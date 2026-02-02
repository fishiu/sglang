import argparse
import itertools
import os
import sys
from typing import Any, Dict, List

METHOD_CHOICES = ("base", "quest", "stream")
DEFAULT_WORKDIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(DEFAULT_WORKDIR, "bench_run.sh")

MODEL_MAPPING = {
    "qw_mha_1b": "/iopsstor/scratch/cscs/xjin/cache/hf_home/hub/models--Qwen--Qwen1.5-4B-Chat/snapshots/a7a4d4945d28bac955554c9abd2f74a71ebbf22f",
    "qw_4b": "/iopsstor/scratch/cscs/xjin/cache/hf_home/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554",
    "qw_4b_th": "/iopsstor/scratch/cscs/xjin/cache/hf_home/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/768f209d9ea81521153ed38c47d515654e938aea",
    "qw_30b_th": "/iopsstor/scratch/cscs/xjin/cache/hf_home/hub/models--Qwen--Qwen3-30B-A3B-Thinking-2507/snapshots/144afc2f379b542fdd4e85a1fcd5e1f79112d95d",
    "qw_30b": "/iopsstor/scratch/cscs/xjin/cache/hf_home/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
}


def parse_range(expr: str) -> List[int]:
    values: List[int] = []
    for part in expr.split(","):
        token = part.strip()
        if not token:
            continue
        if ":" in token:
            segments = token.split(":")
            if len(segments) != 3:
                raise ValueError(f"Invalid range '{token}'. Expected start:end:step.")
            start, end, step = (int(x) for x in segments)
            values.extend(range(start, end + 1, step))
        else:
            values.append(int(token))
    if not values:
        raise ValueError(f"No numeric values parsed from '{expr}'.")
    return values


def parse_float_list(expr: str) -> List[float]:
    values: List[float] = []
    for part in expr.split(","):
        token = part.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError(f"No float values parsed from '{expr}'.")
    return values


def parse_str_list(expr: str) -> List[str]:
    items = [token.strip() for token in expr.split(",") if token.strip()]
    if not items:
        raise ValueError(f"No string values parsed from '{expr}'.")
    return items


def parse_ctx(expr: str) -> List[str]:
    return [token.strip() for token in expr.split(",") if token.strip()]


def parse_methods(expr: str) -> List[str]:
    methods = [token.strip().lower() for token in expr.split(",") if token.strip()]
    if not methods:
        raise ValueError("At least one method must be provided.")
    invalid = sorted(set(m for m in methods if m not in METHOD_CHOICES))
    if invalid:
        raise ValueError(f"Unsupported method(s): {invalid}. Allowed: {METHOD_CHOICES}")
    return methods


def resolve_model_entry(key: str) -> Dict[str, str]:
    path = MODEL_MAPPING.get(key, key)
    label_source = key if key in MODEL_MAPPING else os.path.basename(path.rstrip("/")) or key
    return {"key": key, "path": path, "label": sanitize(label_source)}


def sanitize(value: str) -> str:
    sanitized = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(value))
    sanitized = sanitized.strip("_")
    return sanitized or "value"


def format_float(value: float) -> str:
    return format(value, "g")


def main():
    parser = argparse.ArgumentParser(description="Generate SLURM scripts for bench_one_batch sweeps.")
    parser.add_argument("--model", "--models", required=True, dest="models", help="Comma separated list of model keys or HF paths.")
    parser.add_argument("--method", "--methods", default="base", dest="methods", help="Comma separated list of sparsity methods (base, quest, stream).")
    parser.add_argument("--bsz", "--batch-size", default="64", dest="batch_size", help="Batch size options (int list or start:end:step).")
    parser.add_argument("--in", "--input-len", default="1000", dest="input_len", help="Input length options (int list or start:end:step).")
    parser.add_argument("--out", "--output-len", default="10", dest="output_len", help="Output length options (int list or start:end:step).")
    parser.add_argument("--pg", "--page-size", default="16", dest="page_size", help="Page size options (int list or start:end:step).")
    parser.add_argument("--mem", "--mem-frac", default="0.5", dest="mem_frac", help="mem_fraction_static options (float list).")
    parser.add_argument("--ctx", default="none", help="Context length options (comma separated, use 'none' to skip).")
    parser.add_argument("--topk", default=None, help="quest_topk options (range or list). Only valid with quest; defaults to 16 when omitted.")
    parser.add_argument("--est", default=None, help="quest_estimate_kernel options (comma list). Only valid with quest; defaults to triton when omitted.")
    parser.add_argument("--wind", default=None, help="streaming_llm_window_length options (range or list). Only valid with stream; defaults to 512 when omitted.")
    parser.add_argument("--sink", default=None, help="streaming_llm_num_sink_tokens options (range or list). Only valid with stream; defaults to 4 when omitted.")
    parser.add_argument("--output", required=True, help="Output shell script path.")
    parser.add_argument("--array", type=int, default=10, help="Maximum number of concurrent jobs in the SLURM array.")
    parser.add_argument("--name", required=True, help="Directory name for storing run logs (e.g., 20251205/quest).")
    parser.add_argument("--account", default="a-g200", help="SLURM account.")
    parser.add_argument("--partition", default="normal", help="SLURM partition name.")
    parser.add_argument("--time", default="0:20:00", help="SLURM time limit.")
    parser.add_argument("--conda-env", default="sgl", help="Conda environment to activate inside the script.")
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR, help="Working directory that runs the benchmarks.")

    args = parser.parse_args()

    try:
        models = [resolve_model_entry(m) for m in parse_str_list(args.models)]
        methods = parse_methods(args.methods)
        batch_sizes = parse_range(args.batch_size)
        input_lens = parse_range(args.input_len)
        output_lens = parse_range(args.output_len)
        page_sizes = parse_range(args.page_size)
        mem_fracs = parse_float_list(args.mem_frac)
        ctxs = parse_ctx(args.ctx)
        ctxs = ctxs or ["none"]
    except ValueError as exc:
        print(f"Error parsing arguments: {exc}")
        sys.exit(1)

    quest_topks: List[int] = []
    quest_ests: List[str] = []
    quest_args_supplied = args.topk is not None or args.est is not None
    if "quest" in methods:
        topk_expr = args.topk if args.topk is not None else "16"
        est_expr = args.est if args.est is not None else "triton"
        try:
            quest_topks = parse_range(topk_expr)
            quest_ests = parse_str_list(est_expr)
        except ValueError as exc:
            print(f"Error parsing Quest args: {exc}")
            sys.exit(1)
    elif quest_args_supplied:
        print("Error: --topk and --est are only allowed when quest is included in --method.")
        sys.exit(1)

    stream_winds: List[int] = []
    stream_sinks: List[int] = []
    stream_args_supplied = args.wind is not None or args.sink is not None
    if "stream" in methods:
        wind_expr = args.wind if args.wind is not None else "512"
        sink_expr = args.sink if args.sink is not None else "4"
        try:
            stream_winds = parse_range(wind_expr)
            stream_sinks = parse_range(sink_expr)
        except ValueError as exc:
            print(f"Error parsing StreamingLLM args: {exc}")
            sys.exit(1)
    elif stream_args_supplied:
        print("Error: --wind and --sink are only valid when stream is included in --method.")
        sys.exit(1)

    def method_specific_configs(method: str) -> List[Dict[str, Any]]:
        if method == "quest":
            return [{"topk": topk, "est": est} for topk, est in itertools.product(quest_topks, quest_ests)]
        if method == "stream":
            return [{"wind": wind, "sink": sink} for wind, sink in itertools.product(stream_winds, stream_sinks)]
        return [{}]

    experiments = []
    task_counter = 0

    for model_entry in models:
        for bsz in batch_sizes:
            for inp in input_lens:
                for out in output_lens:
                    for page in page_sizes:
                        for frac in mem_fracs:
                            for ctx in ctxs:
                                for method in methods:
                                    for spec in method_specific_configs(method):
                                        task_counter += 1
                                        cmd = build_command(
                                            model_entry["path"],
                                            bsz,
                                            inp,
                                            out,
                                            page,
                                            frac,
                                            ctx,
                                            method,
                                            spec,
                                        )
                                        log_file = build_log_filename(
                                            task_counter,
                                            model_entry["label"],
                                            bsz,
                                            inp,
                                            out,
                                            page,
                                            frac,
                                            ctx,
                                            method,
                                            spec,
                                        )
                                        experiments.append(
                                            {
                                                "cmd": cmd,
                                                "log_file": log_file,
                                                "method": method,
                                                "spec": spec,
                                                "model_key": model_entry["key"],
                                            }
                                        )

    if not experiments:
        print("No experiments generated. Please check your argument combinations.")
        sys.exit(1)

    print("Configuration:")
    print(f"  Models: {[entry['key'] for entry in models]}")
    print(f"  Methods: {methods}")
    print(f"  BatchSizes: {batch_sizes}")
    print(f"  InputLens: {input_lens}")
    print(f"  OutputLens: {output_lens}")
    print(f"  PageSizes: {page_sizes}")
    print(f"  Fractions: {mem_fracs}")
    print(f"  Contexts: {ctxs}")
    if "quest" in methods:
        print(f"  Quest TopKs: {quest_topks}")
        print(f"  Quest Estimate Kernels: {quest_ests}")
    if "stream" in methods:
        print(f"  Stream Windows: {stream_winds}")
        print(f"  Stream Sink Tokens: {stream_sinks}")
    print(f"  Total experiments: {len(experiments)}")
    print("-" * 40)

    progress_log_path = os.path.join(args.name, "progress.log")

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write("#!/bin/bash\n\n")
            f.write(f"#SBATCH --account={args.account}\n")
            f.write(f"#SBATCH --job-name=bench_one_batch\n")
            f.write("#SBATCH --output=/dev/null\n")
            f.write("#SBATCH --error=/dev/null\n")
            f.write(f"#SBATCH --partition={args.partition}\n")
            f.write(f"#SBATCH --time={args.time}\n")
            f.write(f"#SBATCH --array=1-{len(experiments)}%{args.array}\n\n")

            f.write("# Generated by bench_gen.py\n")
            f.write("# Configuration summary:\n")
            f.write(f"#   Models: {[entry['key'] for entry in models]}\n")
            f.write(f"#   Methods: {methods}\n")
            f.write(f"#   BatchSizes: {batch_sizes}\n")
            f.write(f"#   InputLens: {input_lens}\n")
            f.write(f"#   OutputLens: {output_lens}\n")
            f.write(f"#   PageSizes: {page_sizes}\n")
            f.write(f"#   Fractions: {mem_fracs}\n")
            f.write(f"#   Contexts: {ctxs}\n")
            f.write(f"#   Array limit: {args.array}\n")
            f.write(f"#   Log Directory: {args.name}\n")
            if "quest" in methods:
                f.write(f"#   Quest TopKs: {quest_topks}\n")
                f.write(f"#   Quest Estimate Kernels: {quest_ests}\n")
            if "stream" in methods:
                f.write(f"#   Stream Windows: {stream_winds}\n")
                f.write(f"#   Stream Sink Tokens: {stream_sinks}\n")
            f.write("\n")

            f.write("source /iopsstor/scratch/cscs/xjin/miniconda3/etc/profile.d/conda.sh\n")
            f.write(f"conda activate {args.conda_env}\n")
            f.write(f"cd {args.workdir}\n")
            f.write(f"mkdir -p {args.name}\n\n")

            f.write("case $SLURM_ARRAY_TASK_ID in\n")
            for idx, exp in enumerate(experiments, start=1):
                log_path = os.path.join(args.name, exp["log_file"])
                f.write(f"    {idx})\n")
                f.write(f"        echo 'Running experiment {idx}/{len(experiments)} -> {exp['log_file']}'\n")
                f.write("        start_time=$(date +%s)\n")
                f.write(f"        echo 'Command: {exp['cmd']}' > \"{log_path}\"\n")
                f.write(f"        echo \"Start time: $(date)\" >> \"{log_path}\"\n")
                f.write(f"        {exp['cmd']} >> \"{log_path}\" 2>&1\n")
                f.write("        exit_code=$?\n")
                f.write("        end_time=$(date +%s)\n")
                f.write(f"        echo \"End time: $(date)\" >> \"{log_path}\"\n")
                f.write("        duration=$((end_time - start_time))\n")
                f.write("        hours=$((duration / 3600))\n")
                f.write("        minutes=$(((duration % 3600) / 60))\n")
                f.write("        seconds=$((duration % 60))\n")
                f.write(f"        echo \"Duration: ${{hours}}h ${{minutes}}m ${{seconds}}s\" >> \"{log_path}\"\n")
                f.write(f"        echo \"Completed $(pwd)/{log_path} ({idx}/{len(experiments)}) Exit Code: $exit_code\" >> \"{progress_log_path}\"\n")
                f.write("        ;;\n")
            f.write("esac\n")

        print(f"Successfully generated {len(experiments)} experiments in '{args.output}'.")
    except OSError as exc:
        print(f"Error while writing {args.output}: {exc}")
        sys.exit(1)


def build_command(
    model_path: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    page_size: int,
    mem_frac: float,
    ctx: str,
    method: str,
    spec: Dict[str, Any],
) -> str:
    parts = [
        "python -m sglang.bench_one_batch",
        f"--model-path {model_path}",
        "--load-format dummy",
        f"--batch-size {batch_size}",
        f"--input-len {input_len}",
        f"--output-len {output_len}",
        "--attention-backend triton",
        f"--page-size {page_size}",
        f"--mem-fraction-static {format_float(mem_frac)}",
    ]
    if ctx != "none":
        parts.append(f"--context-length {ctx}")

    if method == "quest":
        parts.append("--enable-quest")
        parts.append(f"--quest-topk {spec['topk']}")
        parts.append(f"--quest-estimate-kernel {spec['est']}")
    elif method == "stream":
        parts.append("--enable-streaming-llm")
        parts.append(f"--streaming-llm-window-length {spec['wind']}")
        parts.append(f"--streaming-llm-num-sink-tokens {spec['sink']}")

    return " ".join(parts)


def build_log_filename(
    task_id: int,
    model_label: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    page_size: int,
    mem_frac: float,
    ctx: str,
    method: str,
    spec: Dict[str, Any],
) -> str:
    parts = [
        f"method_{sanitize(method)}",
        f"model_{sanitize(model_label)}",
        f"bsz_{batch_size}",
        f"in_{input_len}",
        f"out_{output_len}",
        f"pg_{page_size}",
        f"frac_{sanitize(format_float(mem_frac))}",
        f"ctx_{sanitize(ctx)}",
    ]
    if method == "quest":
        parts.append(f"topk_{spec['topk']}")
        parts.append(f"est_{sanitize(spec['est'])}")
    elif method == "stream":
        parts.append(f"wind_{spec['wind']}")
        parts.append(f"sink_{spec['sink']}")
    parts.append(f"id_{task_id}")
    return "-".join(parts) + ".log"


if __name__ == "__main__":
    main()
