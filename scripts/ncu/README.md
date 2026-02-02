# NCU 性能分析工具

基于 NVIDIA Nsight Compute Python API 的性能报告解析和 Roofline 分析工具。

## 文件说明

| 文件 | 描述 |
|------|------|
| `ncu_parser.py` | NCU 报告解析器，提取关键性能指标 |
| `plot_roofline_api.py` | Roofline 绘图工具 (基于 NCU Python API) |
| `ncu_report.py` | NCU CSV 导出解析工具 (基于 ncu CLI) |
| `parse_roofline.py` | 旧版 Roofline 解析工具 (基于 ncu CLI CSV) |
| `plot_roofline.py` | 旧版 Roofline 绘图工具 |

## 环境配置

```bash
# 激活 conda 环境
source /iopsstor/scratch/cscs/xjin/miniconda3/etc/profile.d/conda.sh
conda activate sgl

# NCU Python API 路径会自动检测，如需手动设置：
export PYTHONPATH=/path/to/cuda/nsight-compute-xxxx/extras/python:$PYTHONPATH
```

## 使用方法

### 1. 解析 NCU 报告

```bash
python ncu_parser.py --report <ncu-rep文件> --output <输出JSON>

# 示例
python ncu_parser.py \
  --report /path/to/report.ncu-rep \
  --output /path/to/output.json

# 调试：导出所有 metrics
python ncu_parser.py \
  --report /path/to/report.ncu-rep \
  --dump-metrics \
  --kernel-index 0 \
  --output /path/to/all_metrics.json
```

### 2. 绘制 Roofline 图

```bash
python plot_roofline_api.py \
  --report <ncu-rep文件> \
  --output <输出图片> \
  [--title <图标题>] \
  [--json-output <JSON输出>] \
  [--verbose] \
  [--show-labels]

# 示例
python plot_roofline_api.py \
  --report /path/to/report.ncu-rep \
  --output /path/to/roofline.png \
  --json-output /path/to/data.json \
  --verbose \
  --title "My Kernel Roofline Analysis"
```

## Roofline 模型说明

### 标准 Roofline (DRAM + FP32)

本工具生成的是标准的 Roofline 模型，其中：

- **X 轴**: 算术强度 (Arithmetic Intensity, AI) = FLOP / Byte
- **Y 轴**: 性能 (Performance) = FLOP/s
- **屋顶线**: 由内存带宽限制的斜线和计算能力限制的水平线组成
- **脊点 (Ridge Point)**: 斜线和水平线的交点，AI = Peak_FLOP/s / Peak_BW

### 计算公式

```
# 峰值 FP32 性能 (FLOP/s)
Peak_FP32 = 2 × peak_FFMA_per_cycle × SM_frequency

# 峰值 DRAM 带宽 (bytes/s)
Peak_DRAM_BW = memory_clock × bus_width × 2 (DDR)

# 实际 FP32 性能 (FLOP/s)
Achieved_FP32 = (FADD/cycle + FMUL/cycle + 2×FFMA/cycle) × SM_frequency

# 算术强度 (FLOP/Byte)
AI = Total_FLOPs / Total_DRAM_Traffic
   = (FLOP/cycle × cycles) / (bytes_read + bytes_write)

# 脊点
Ridge_Point = Peak_FP32 / Peak_DRAM_BW
```

## 与 NCU GUI 对比验证

工具会输出详细的验证数据，可与 Nsight Compute GUI 中以下 Section 对比：

### Speed of Light Section

| NCU GUI 指标 | 对应 Metric 名称 |
|-------------|-----------------|
| SM Frequency | `sm__cycles_elapsed.avg.per_second` |
| GPU Time | `gpu__time_duration.sum` |
| Memory Throughput | `dram__bytes_read.sum.per_second`, `dram__bytes_write.sum.per_second` |

### GPU Speed Of Light Roofline Chart

| 指标 | Metric 名称 |
|-----|------------|
| DRAM Bandwidth | `dram__bytes.sum.per_second` |
| Peak DRAM BW | `dram__bytes.sum.peak_sustained` × frequency |
| FP32 Operations | `smsp__sass_thread_inst_executed_op_f*.sum.per_cycle_elapsed` |
| Peak FP32 | `sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained` × 2 |

### 关键验证点

1. **执行时间**: `gpu__time_duration.sum` (单位: ns)
2. **DRAM 流量**: `dram__bytes_read.sum` + `dram__bytes_write.sum`
3. **FP32 指令**: 
   - FADD: `smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed`
   - FMUL: `smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed`
   - FFMA: `smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed`

## 输出示例

对于 `on_b128_in1024_out16_w256_s4.ncu-rep`:

```
设备: NVIDIA GH200 120GB
计算能力: 9.0
SM 数量: 132

峰值 FP32 性能: 48.31 TFLOP/s
峰值 DRAM 带宽: 4.02 TB/s
脊点: 12.01 FLOP/Byte

Kernel: _fwd_kernel_stage1_layer0
  - 算术强度: 0.9576 FLOP/Byte
  - 实际性能: 1557.82 GFLOP/s
  - 峰值利用率: 3.22%
  - 状态: Memory-bound (内存受限)
```

## 批量分析示例

```python
import os
from ncu_parser import NcuReportParser

# 批量处理多个报告
reports = [
    "on_b16_in1024_out16_w256_s4.ncu-rep",
    "on_b32_in1024_out16_w256_s4.ncu-rep", 
    "on_b64_in1024_out16_w256_s4.ncu-rep",
    "on_b128_in1024_out16_w256_s4.ncu-rep",
]

results = []
for report in reports:
    parser = NcuReportParser(report).load()
    summary = parser.get_summary()
    results.append(summary)

# 分析结果...
```

## 参考资料

- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [Roofline Model - Wikipedia](https://en.wikipedia.org/wiki/Roofline_model)
- [Kernel Profiling Guide - Roofline](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline)
