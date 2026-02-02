import argparse
import itertools
import sys
import os

def parse_range(s):
    # handle 1:5:2 -> 1, 3, 5 (inclusive end)
    # also handle single integer case
    if ':' in s:
        parts = [int(x) for x in s.split(':')]
        start, end, step = parts[0], parts[1], parts[2]
        return list(range(start, end + 1, step))
    else:
        return [int(x) for x in s.split(',')]

def parse_float_list(s):
    return [x if x == 'none' else float(x) for x in s.split(',')]

def parse_str_list(s):
    return s.split(',')

def parse_bsz(s):
    return s.split(',')

def parse_ctx(s):
    return s.split(',')

def parse_toks(s):
    return [x if x == 'none' else int(x) for x in s.split(',')]

MODEL_MAPPING = {
    'qw_mha_1b': '/iopsstor/scratch/cscs/xjin/cache/hf_home/hub/models--Qwen--Qwen1.5-4B-Chat/snapshots/a7a4d4945d28bac955554c9abd2f74a71ebbf22f',
    'qw_4b': '/iopsstor/scratch/cscs/xjin/cache/hf_home/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554',
    'qw_4b_th': '/iopsstor/scratch/cscs/xjin/cache/hf_home/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/768f209d9ea81521153ed38c47d515654e938aea',
    'qw_30b_th': '/iopsstor/scratch/cscs/xjin/cache/hf_home/hub/models--Qwen--Qwen3-30B-A3B-Thinking-2507/snapshots/144afc2f379b542fdd4e85a1fcd5e1f79112d95d',
    'qw_30b': '/iopsstor/scratch/cscs/xjin/cache/hf_home/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe'
}

def main():
    parser = argparse.ArgumentParser(description="Generate shell script for lm_eval experiments")
    parser.add_argument('--model', required=True, help='Comma separated list of model keys (qw_mha_1_5b, qw_4b, qw_4b_th, qw_30b_th, qw_30b)')
    parser.add_argument('--method', required=True, help='Comma separated list of methods (base, quest, stream)')
    
    # Quest parameters
    parser.add_argument('--topk', default='16', help='quest_topk options (int list or range start:end:step). REQUIRED for quest.')
    parser.add_argument('--est', default='triton', help='quest_estimate_kernel options (comma separated string). OPTIONAL for quest.')
    
    # StreamingLLM parameters
    parser.add_argument('--wind', default='256', help='streaming_llm_window_length options (int list or range). REQUIRED for stream.')
    parser.add_argument('--sink', default='4', help='streaming_llm_num_sink_tokens options (int list or range). REQUIRED for stream.')

    # Common parameters
    parser.add_argument('--pg', default='16', help='page_size options (int list or range start:end:step)')
    parser.add_argument('--graph', default='1', help='enable_cuda_graph options (0 or 1, comma separated)')
    parser.add_argument('--frac', default='0.5', help='mem_fraction_static options (float list)')
    parser.add_argument('--limit', default='1', help='limit options (float list)')
    parser.add_argument('--bsz', default='auto', help='batch_size options (auto or int list)')
    parser.add_argument('--ctx', default='none', help='context_length options (none or int list)')
    parser.add_argument('--toks', default='none', help='max_gen_toks options (none or int list)')
    parser.add_argument('--tasks', default='mmlu', help='Comma separated list of tasks (mmlu, longbench, ruler, aime)')
    parser.add_argument('--run', default='1', help='Run IDs (int list or range start:end:step). e.g. 1,2 or 1:3:1')
    parser.add_argument('--tp', type=int, default=1, help='Tensor Parallelism Size (int)')
    parser.add_argument('--output', default='run_eval.sh', help='Output shell script file path')
    parser.add_argument('--array', type=int, default=10, help='Maximum number of parallel jobs for SLURM array')
    parser.add_argument('--log_level', default='info', help='log_level options (comma separated string)')
    parser.add_argument('--dump', action='store_true', help='Enable output dumping to json')
    parser.add_argument('--tpl', action='store_true', help='Pass --apply_chat_template to lm_eval')
    parser.add_argument('--name', required=True, help='Directory name for storing logs (e.g., 20251128/ruler)')

    args = parser.parse_args()

    # Parse arguments
    try:
        models = parse_str_list(args.model)
        methods = parse_str_list(args.method)
        pgs = parse_range(args.pg)
        graphs = parse_range(args.graph)
        fracs = parse_float_list(args.frac)
        limits = parse_float_list(args.limit)
        bszs = parse_bsz(args.bsz)
        ctxs = parse_ctx(args.ctx)
        toks = parse_toks(args.toks)
        tasks = parse_str_list(args.tasks)
        runs = parse_range(args.run)
        
        topks = [None]
        ests = [None]
        winds = [None]
        sinks = [None]
        
        # Always parse these, we will only use them if method matches
        topks = parse_range(args.topk)
        ests = parse_str_list(args.est)
        winds = parse_range(args.wind)
        sinks = parse_range(args.sink)
            
    except ValueError as e:
        print(f"Error parsing arguments: {e}")
        sys.exit(1)

    # Debug info
    print("Configuration:")
    print(f"  Methods: {methods}")
    print(f"  Models: {models}")
    print(f"  PageSizes: {pgs}")
    print(f"  Graphs: {graphs}")
    print(f"  Fractions: {fracs}")
    print(f"  Limits: {limits}")
    print(f"  BatchSizes: {bszs}")
    print(f"  ContextLengths: {ctxs}")
    print(f"  Tasks: {tasks}")
    print(f"  LogLevel: {args.log_level}")
    print(f"  TP Size: {args.tp}")
    print(f"  Runs: {runs}")
    print(f"  Max Parallel Jobs: {args.array}")
    print(f"  Dump Output: {args.dump}")
    print(f"  Tpl: {args.tpl}")
    print(f"  Log Directory: {args.name}")
    
    if 'quest' in methods:
        print(f"  Quest TopKs: {topks}")
        print(f"  Quest Estimates: {ests}")
    if 'stream' in methods:
        print(f"  Stream Windows: {winds}")
        print(f"  Stream Sinks: {sinks}")

    print("-" * 40)

    experiments = []

    for method in methods:
        # Iterate over all combinations
        keys = ['model', 'pg', 'graph', 'frac', 'limit', 'bsz', 'ctx', 'toks', 'tasks', 'run']
        iterables = [models, pgs, graphs, fracs, limits, bszs, ctxs, toks, tasks, runs]
        
        if method == 'quest':
            keys.extend(['topk', 'est'])
            iterables.extend([topks, ests])
        elif method == 'stream':
            keys.extend(['wind', 'sink'])
            iterables.extend([winds, sinks])

        combinations = itertools.product(*iterables)

        for combo in combinations:
            c = dict(zip(keys, combo))

            # Validate model
            if c['model'] not in MODEL_MAPPING:
                print(f"Warning: Unknown model code '{c['model']}', skipping.")
                continue
            
            real_model_name = MODEL_MAPPING[c['model']]

            # Construct model_args
            ma = [
                f"pretrained={real_model_name}",
                "dp_size=1",
                f"tp_size={args.tp}",
                "dtype=auto",
                f"mem_fraction_static={c['frac']}",
                f"page_size={c['pg']}",
                f"log_level={args.log_level}",
                "enable_nan_detection=True",
                "attention_backend=triton"
            ]
            
            if c['ctx'] != 'none':
                ma.append(f"context_length={c['ctx']}")

            if method == 'quest':
                ma.append("enable_quest=True")
                ma.append(f"quest_topk={c['topk']}")
                ma.append(f"quest_estimate_kernel={c['est']}")
            elif method == 'stream':
                ma.append("enable_streaming_llm=True")
                ma.append(f"streaming_llm_window_length={c['wind']}")
                ma.append(f"streaming_llm_num_sink_tokens={c['sink']}")

            # Handle graph (0 -> disable, 1 -> default/enable)
            if c['graph'] == 0:
                ma.append("disable_cuda_graph=True")
            
            model_args_str = ",".join(ma)

            # Construct full command
            if c['limit'] == 'none':
                base_cmd = f"lm_eval --model sglang --model_args {model_args_str} --tasks {c['tasks']} --batch_size {c['bsz']}"
            else:
                base_cmd = f"lm_eval --model sglang --model_args {model_args_str} --tasks {c['tasks']} --batch_size {c['bsz']} --limit {c['limit']}"
            
            if c['toks'] != 'none':
                base_cmd += f" --gen_kwargs '{{\"max_gen_toks\": {c['toks']}}}'"

            if args.tpl:
                base_cmd += " --apply_chat_template"

            # Construct log filename
            # parts common to all
            log_parts = [
                f"method_{method}",
                f"model_{c['model']}",
                f"pg_{c['pg']}",
                f"graph_{c['graph']}",
                f"frac_{c['frac']}",
                f"limit_{c['limit']}",
                f"bsz_{c['bsz']}",
                f"ctx_{c['ctx']}",
                f"toks_{c['toks']}",
                f"tasks_{c['tasks']}"
            ]
            
            if method == 'quest':
                log_parts.append(f"topk_{c['topk']}")
                log_parts.append(f"est_{c['est']}")
            elif method == 'stream':
                log_parts.append(f"wind_{c['wind']}")
                log_parts.append(f"sink_{c['sink']}")
                
            log_parts.append(f"run_{c['run']}.log")
            
            log_filename = "-".join(log_parts)
            
            # Add output dump path if enabled
            if args.dump:
                json_filename = log_filename.replace('.log', '.json')
                # Using the log directory for the json output
                json_path = os.path.join(args.name, json_filename)
                base_cmd += f" --log_samples --output_path {json_path}"

            experiments.append({
                'cmd': base_cmd,
                'log_file': log_filename
            })

    # Generate File
    try:
        with open(args.output, 'w') as f:
            f.write("#!/bin/bash\n\n")
            
            # SBATCH directives
            # Extract base name of the log directory for job name
            job_name = args.name
            f.write(f"#SBATCH --account=a-g200\n")
            f.write(f"#SBATCH --job-name={job_name}\n")
            # f.write(f"#SBATCH --output={args.name}/job_%a.out\n")
            f.write(f"#SBATCH --output=/dev/null\n")
            # f.write(f"#SBATCH --error={args.name}/job_%a.err\n")
            f.write(            f"#SBATCH --error=/dev/null\n")
            f.write("#SBATCH --partition=normal\n")
            f.write("#SBATCH --time=12:00:00\n")
            f.write(f"#SBATCH --gpus={args.tp}\n")
            
            # Determine number of tasks
            num_experiments = len(experiments)
            f.write(f"#SBATCH --array=1-{num_experiments}%{args.array}\n\n")

            f.write("# Generated by generator.py\n")
            f.write("# Configuration:\n")
            f.write(f"#   Methods: {methods}\n")
            f.write(f"#   Models: {models}\n")
            f.write(f"#   PageSizes: {pgs}\n")
            f.write(f"#   Graphs: {graphs}\n")
            f.write(f"#   Fractions: {fracs}\n")
            f.write(f"#   Limits: {limits}\n")
            f.write(f"#   BatchSizes: {bszs}\n")
            f.write(f"#   ContextLengths: {ctxs}\n")
            f.write(f"#   Tasks: {tasks}\n")
            f.write(f"#   TP Size: {args.tp}\n")
            f.write(f"#   Runs: {runs}\n")
            f.write(f"#   Max Parallel Jobs: {args.array}\n")
            f.write(f"#   Log Directory: {args.name}\n")
            f.write(f"#   Tpl: {args.tpl}\n")
            if 'quest' in methods:
                 f.write(f"#   Quest TopKs: {topks}\n")
                 f.write(f"#   Quest Estimates: {ests}\n")
            if 'stream' in methods:
                 f.write(f"#   Stream Windows: {winds}\n")
                 f.write(f"#   Stream Sinks: {sinks}\n")
            f.write("\n")
            
            f.write("source /iopsstor/scratch/cscs/xjin/miniconda3/etc/profile.d/conda.sh\n")
            f.write("conda activate sgl\n")
            f.write("cd /iopsstor/scratch/cscs/xjin/repos/sglang/workspace/exp/harness\n")
            f.write(f"mkdir -p {args.name}\n\n")
            
            f.write("# Array job execution\n")
            f.write("case $SLURM_ARRAY_TASK_ID in\n")
            
            for idx, exp in enumerate(experiments):
                task_id = idx + 1
                f.write(f"    {task_id})\n")
                
                cmd = exp['cmd']
                log_file_path = os.path.join(args.name, exp['log_file'])
                
                f.write(f"        echo 'Running experiment: {exp['log_file']}'\n")
                f.write(f"        echo \"Start time: $(date)\"\n")
                f.write("        start_time=$(date +%s)\n")
                
                # Write command to log file first
                f.write(f"        echo 'Command: {cmd}' > {log_file_path}\n")
                f.write(f"        echo \"Start time: $(date)\" >> {log_file_path}\n")
                
                # Execute and redirect output to log file
                f.write(f"        {cmd} >> {log_file_path} 2>&1\n")
                f.write("        exit_code=$?\n")
                
                f.write("        end_time=$(date +%s)\n")
                f.write(f"        echo \"End time: $(date)\"\n")
                f.write(f"        echo \"End time: $(date)\" >> {log_file_path}\n")
                
                # Calculate duration
                f.write("        duration=$((end_time - start_time))\n")
                f.write("        hours=$((duration / 3600))\n")
                f.write("        minutes=$(( (duration % 3600) / 60 ))\n")
                f.write("        seconds=$((duration % 60))\n")
                
                f.write(f"        echo \"Duration: ${{hours}}h ${{minutes}}m ${{seconds}}s\"\n")
                f.write(f"        echo \"Duration: ${{hours}}h ${{minutes}}m ${{seconds}}s\" >> {log_file_path}\n")
                f.write(f"        echo \"Tpl: {int(args.tpl)}\" >> {log_file_path}\n")
                
                # Progress log
                progress_log_path = os.path.join(args.name, "progress.log")
                progress = f"{task_id}/{len(experiments)}"
                f.write(f"        echo \"Completed $(pwd)/{log_file_path} ({progress}) Exit Code: $exit_code\" >> {progress_log_path}\n")
                f.write("        ;;\n")
                
            f.write("esac\n")
        
        print(f"Successfully generated {len(experiments)} experiments in '{args.output}'.")
        
    except IOError as e:
        print(f"Error writing to file {args.output}: {e}")

if __name__ == "__main__":
    main()
