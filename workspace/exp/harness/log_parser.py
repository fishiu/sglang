import os
import re
import glob

class ExperimentInfo:
    def __init__(self, log_path):
        self.log_path = log_path
        self.filename = os.path.basename(log_path)
        self.params = {}
        self.parse_filename()
        self.score = None
        
    def parse_filename(self):
        # Remove .log extension
        name = self.filename.rsplit('.', 1)[0]
        parts = name.split('-')
        
        # known keys from generator.py
        # method, model, pg, graph, frac, limit, bsz, ctx, tasks, run
        # quest: topk, est
        # stream: wind, sink
        
        for p in parts:
            if '_' in p:
                key, value = p.split('_', 1)
                self.params[key] = value
                
        self.method = self.params.get('method')
        self.model = self.params.get('model')
        self.task = self.params.get('tasks')
        self.pg = self.params.get('pg')
        self.graph = self.params.get('graph')
        self.frac = self.params.get('frac')
        self.limit = self.params.get('limit')
        self.bsz = self.params.get('bsz')
        self.ctx = self.params.get('ctx')
        self.run = self.params.get('run')

    def extract_score(self):
        raise NotImplementedError("Subclasses must implement extract_score")
        
    def read_last_lines(self, n=200):
        """Read last n lines of the file."""
        try:
            with open(self.log_path, 'rb') as f:
                f.seek(0, os.SEEK_END)
                filesize = f.tell()
                block_size = 1024
                data = b''
                lines_found = 0
                
                offset = filesize
                while offset > 0 and lines_found < n:
                    read_size = min(block_size, offset)
                    offset -= read_size
                    f.seek(offset)
                    chunk = f.read(read_size)
                    data = chunk + data
                    lines_found = data.count(b'\n')
                    
                lines = data.decode('utf-8', errors='ignore').splitlines()
                return lines[-n:]
        except Exception as e:
            print(f"Error reading {self.log_path}: {e}")
            return []

class LongBenchInfo(ExperimentInfo):
    def extract_score(self):
        # Look for Groups table and average Value
        lines = self.read_last_lines(300)
        in_groups_table = False
        values = []
        
        for line in lines:
            if "|       Groups        |" in line:
                in_groups_table = True
                continue
            
            if in_groups_table:
                if line.strip() == "" or "Traceback" in line or "End time" in line:
                    in_groups_table = False
                    continue
                
                # | - Code Completion   |      0|none  |      |score |↑  |0.1930|±  |0.0071|
                if "|" in line and "score" in line:
                    parts = [p.strip() for p in line.split('|')]
                    try:
                        val_str = parts[7]
                        val = float(val_str)
                        values.append(val)
                    except (ValueError, IndexError):
                        pass
                        
        if values:
            self.score = sum(values) / len(values)
        return self.score

class MMLUInfo(ExperimentInfo):
    def extract_score(self):
        # Look for |mmlu | ... | acc | ... | Value |
        lines = self.read_last_lines(200)
        for line in lines:
            if "|mmlu" in line and "acc" in line:
                parts = [p.strip() for p in line.split('|')]
                try:
                    # The value is usually at index 7 or so
                    val_str = parts[7]
                    self.score = float(val_str)
                    return self.score
                except (ValueError, IndexError):
                    pass
        return None

class RulerInfo(ExperimentInfo):
    def extract_score(self):
        # Look for |ruler | ...
        lines = self.read_last_lines(200)
        for line in lines:
            if "|ruler" in line:
                parts = [p.strip() for p in line.split('|')]
                try:
                    val_str = parts[7]
                    self.score = float(val_str)
                    return self.score
                except (ValueError, IndexError):
                    pass
        return None

def get_experiment_info(log_path):
    base_name = os.path.basename(log_path)
    if 'tasks_longbench' in base_name:
        return LongBenchInfo(log_path)
    elif 'tasks_mmlu' in base_name:
        return MMLUInfo(log_path)
    elif 'tasks_ruler' in base_name:
        return RulerInfo(log_path)
    else:
        return ExperimentInfo(log_path)

def get_successful_logs(root_dirs):
    success_logs = set()
    
    for root_dir in root_dirs:
        # Find progress.log files
        for dirpath, _, filenames in os.walk(root_dir):
            if 'progress.log' in filenames:
                progress_path = os.path.join(dirpath, 'progress.log')
                try:
                    with open(progress_path, 'r') as f:
                        for line in f:
                            if "Exit Code: 0" in line:
                                # Extract path
                                # "Completed /abs/path/to/log (x/y) Exit Code: 0"
                                match = re.search(r'Completed (.*?) \(', line)
                                if match:
                                    raw_path = match.group(1)
                                    
                                    # Try to resolve the path relative to current environment
                                    target_suffix = ""
                                    if "/workspace/exp/harness/" in raw_path:
                                        target_suffix = raw_path.split("/workspace/exp/harness/")[1]
                                    else:
                                        # fallback if path structure is different
                                        continue

                                    # Candidates for local path
                                    # 1. CWD is 'workspace/exp/harness' -> target_suffix (e.g. "20251130/...")
                                    # 2. CWD is project root -> "workspace/exp/harness/" + target_suffix
                                    candidates = [
                                        target_suffix,
                                        os.path.join("workspace/exp/harness", target_suffix)
                                    ]
                                    
                                    found = False
                                    for cand in candidates:
                                        if os.path.exists(cand):
                                            success_logs.add(cand)
                                            found = True
                                            break
                                    
                                    if not found:
                                        # If still not found, maybe print a debug warning if verbose
                                        pass
                except Exception as e:
                    print(f"Error reading {progress_path}: {e}")

    return list(success_logs)
