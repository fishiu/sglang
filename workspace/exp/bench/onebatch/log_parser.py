import os
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Union

@dataclass
class Info:
    method: str
    model: str
    bsz: int
    input_len: int
    output_len: int
    page_size: int
    mem_frac: float
    ctx: str
    task_id: int
    
    # Optional fields
    topk: Optional[int] = None
    est: Optional[str] = None
    wind: Optional[int] = None
    sink: Optional[int] = None
    
    # Result
    status: str = "ok" # ok, oom, limit, error
    throughput: Optional[float] = None
    
    log_path: str = ""

    def to_dict(self):
        return self.__dict__

def parse_filename_args(filename: str) -> Optional[Dict]:
    # Remove .log if present
    base = filename
    if base.endswith(".log"):
        base = base[:-4]
        
    # Define regex pattern to match the fixed prefix structure
    # Structure: method_{}-model_{}-bsz_{}-in_{}-out_{}-pg_{}-frac_{}-ctx_{}-...
    
    pattern = (
        r"^method_(?P<method>[^_]+)-"
        r"model_(?P<model>.+?)-"
        r"bsz_(?P<bsz>\d+)-"
        r"in_(?P<input_len>\d+)-"
        r"out_(?P<output_len>\d+)-"
        r"pg_(?P<page_size>\d+)-"
        r"frac_(?P<mem_frac>[\d\.]+)-"
        r"ctx_(?P<ctx>[^-]+)" 
        r"(?P<rest>.*)$"
    )
    
    match = re.match(pattern, base)
    if not match:
        return None
        
    groups = match.groupdict()
    info = {
        "method": groups["method"],
        "model": groups["model"],
        "bsz": int(groups["bsz"]),
        "input_len": int(groups["input_len"]),
        "output_len": int(groups["output_len"]),
        "page_size": int(groups["page_size"]),
        "mem_frac": float(groups["mem_frac"]),
        "ctx": groups["ctx"],
    }
    
    rest = groups["rest"]
    
    # Extract ID from the end
    # Expecting ...-id_{id}
    id_match = re.search(r"-id_(?P<id>\d+)$", rest)
    if not id_match:
        # Check if rest starts with -id_ or just id_ (if rest was just id)
        if rest.startswith("-id_") or rest.startswith("id_"):
             id_match = re.search(r"id_(?P<id>\d+)$", rest)
    
    if not id_match:
        return None
        
    info["task_id"] = int(id_match.group("id"))
    
    # Process the middle part (between ctx and id)
    # rest includes the leading dash if it existed in the pattern match group (which captured .* starting after ctx_)
    # Actually, regex `r"ctx_(?P<ctx>[^-]+)(?P<rest>.*)$"`
    # If filename is ...-ctx_1024-id_1
    # ctx matches 1024
    # rest matches -id_1
    
    # Get the middle segment by removing the id part
    middle = rest[:id_match.start()]
    # Remove leading dash from middle if present
    if middle.startswith("-"):
        middle = middle[1:]
        
    # Parse method-specific args from middle
    if info["method"] == "quest":
        # Expect topk_{}-est_{}
        m_topk = re.search(r"topk_(?P<topk>\d+)", middle)
        m_est = re.search(r"est_(?P<est>[^-]+)", middle)
        if m_topk: info["topk"] = int(m_topk.group("topk"))
        if m_est: info["est"] = m_est.group("est")
        
    elif info["method"] == "stream":
        # Expect wind_{}-sink_{}
        m_wind = re.search(r"wind_(?P<wind>\d+)", middle)
        m_sink = re.search(r"sink_(?P<sink>\d+)", middle)
        if m_wind: info["wind"] = int(m_wind.group("wind"))
        if m_sink: info["sink"] = int(m_sink.group("sink"))
        
    return info

def parse_log_content(content: str) -> Tuple[str, Optional[float]]:
    if "CUDA out of memory" in content:
        return "oom", None
        
    if "skipping" in content and "max batch size limit" in content:
        return "limit", None
    
    # Look for throughput
    # We want the SECOND occurrence.
    # Regex for throughput: "median throughput:   2040.19 token/s"
    matches = re.findall(r"median throughput:\s+([\d\.]+)\s+token/s", content)
    
    if len(matches) >= 2:
        return "ok", float(matches[1])
    
    # If we have matches but less than 2, it might be incomplete or just one run
    # For now, treat as error/incomplete per strict requirements, 
    # but we can check if it finished successfully.
    # The file ends with "End time: ...".
    if "End time:" in content and len(matches) > 0:
        # Fallback to last match if only 1?
        # User said: "Attention must be second because first is warmup".
        # If only 1 exists, maybe warmup didn't print? or only warmup ran?
        return "error", None 
        
    return "error", None

def process_log_file(filepath: str) -> Optional[Info]:
    filename = os.path.basename(filepath)
    if filename == "progress.log":
        return None
    if not filename.endswith(".log"):
        return None
        
    # Parse filename
    args = parse_filename_args(filename)
    if not args:
        # print(f"Failed to parse filename: {filename}")
        return None
        
    # Read content
    try:
        with open(filepath, 'r', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
        
    status, throughput = parse_log_content(content)
    
    info = Info(**args)
    info.status = status
    info.throughput = throughput
    info.log_path = filepath
    
    return info

def get_infos_from_dir(directories: Union[str, List[str]]) -> List[Info]:
    infos = []
    if isinstance(directories, str):
        directories = [directories]
        
    for directory in directories:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file == "progress.log":
                    continue
                if not file.endswith(".log"):
                    continue
                    
                path = os.path.join(root, file)
                info = process_log_file(path)
                if info:
                    infos.append(info)
    return infos

if __name__ == "__main__":
    # Test with the examples provided in the prompt
    
    # Case 1: Success
    f1 = "method_base-model_qw_mha_1b-bsz_16-in_1000-out_10-pg_16-frac_0.62-ctx_1024-id_1.log"
    # Content simulation for testing (in real usage, file must exist)
    # We can rely on the actual files if I run this in the notebook.
    print(f"Testing filename parser on: {f1}")
    print(parse_filename_args(f1))
    
    # Case 2: OOM
    f2 = "method_base-model_qw_4b-bsz_256-in_1000-out_100-pg_16-frac_0.68-ctx_1200-id_190.log"
    print(f"Testing filename parser on: {f2}")
    print(parse_filename_args(f2))
    
    # Case 3: Limit
    f3 = "method_quest-model_qw_mha_1b-bsz_176-in_1000-out_10-pg_16-frac_0.68-ctx_1024-topk_16-est_triton-id_65.log"
    print(f"Testing filename parser on: {f3}")
    print(parse_filename_args(f3))
