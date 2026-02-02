#!/usr/bin/env python3
"""
NCU Parser - 基于 NVIDIA Nsight Compute Python API 的性能报告解析器

用于解析 .ncu-rep 文件，提取关键性能指标，特别是 Roofline 分析所需的数据。

使用前需要设置 PYTHONPATH 包含 NCU 的 Python API 路径，例如：
export PYTHONPATH=/path/to/cuda/nsight-compute-xxxx/extras/python:$PYTHONPATH
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from typing import Any, Dict, List, Optional, Union


def get_ncu_python_path() -> str:
    """尝试自动找到 NCU Python API 路径"""
    # 常见路径
    candidates = [
        "/iopsstor/scratch/cscs/xjin/.local/cuda-12.8/nsight-compute-2025.1.0/extras/python",
        "/usr/local/cuda/nsight-compute/extras/python",
        os.path.expanduser("~/.local/cuda/nsight-compute/extras/python"),
    ]
    for p in candidates:
        if os.path.exists(os.path.join(p, "ncu_report.py")):
            return p
    return ""


# 尝试自动添加路径
_ncu_path = get_ncu_python_path()
if _ncu_path and _ncu_path not in sys.path:
    sys.path.insert(0, _ncu_path)


@dataclasses.dataclass
class DeviceInfo:
    """GPU 设备信息"""
    name: str
    compute_capability: str
    sm_count: int
    gpu_clock_rate_khz: int  # GPU 时钟频率 (kHz)
    memory_clock_rate_khz: int  # 显存时钟频率 (kHz)
    memory_bus_width: int  # 显存总线宽度 (bits)
    
    @property
    def gpu_clock_rate_hz(self) -> float:
        return self.gpu_clock_rate_khz * 1000
    
    @property
    def memory_clock_rate_hz(self) -> float:
        return self.memory_clock_rate_khz * 1000


@dataclasses.dataclass
class RooflineData:
    """单个 kernel 的 Roofline 数据"""
    # 基本信息
    kernel_name: str
    kernel_index: int
    
    # 时间相关 (nanoseconds)
    duration_ns: float
    sm_cycles: float
    sm_frequency_hz: float  # 实际测量的 SM 频率
    
    # DRAM 相关
    dram_bytes_read: float
    dram_bytes_write: float
    dram_bytes_total: float
    dram_peak_bytes_per_cycle: float  # 峰值带宽 (bytes/cycle)
    
    # FP32 计算相关 (指令数/周期)
    fp32_fadd_per_cycle: float
    fp32_fmul_per_cycle: float
    fp32_ffma_per_cycle: float
    fp32_peak_ffma_per_cycle: float  # 峰值 FFMA (inst/cycle)
    
    # FP64 计算相关 (指令数/周期)
    fp64_dadd_per_cycle: float
    fp64_dmul_per_cycle: float
    fp64_dfma_per_cycle: float
    fp64_peak_dfma_per_cycle: float  # 峰值 DFMA (inst/cycle)
    
    # 计算出的 Roofline 指标
    @property
    def achieved_fp32_flops_per_cycle(self) -> float:
        """实际达到的 FP32 FLOP/cycle (FADD + FMUL + 2*FFMA)"""
        return self.fp32_fadd_per_cycle + self.fp32_fmul_per_cycle + 2 * self.fp32_ffma_per_cycle
    
    @property
    def peak_fp32_flops_per_cycle(self) -> float:
        """峰值 FP32 FLOP/cycle (2*peak_FFMA)"""
        return 2 * self.fp32_peak_ffma_per_cycle
    
    @property
    def achieved_fp32_flops_per_second(self) -> float:
        """实际达到的 FP32 性能 (FLOP/s)"""
        return self.achieved_fp32_flops_per_cycle * self.sm_frequency_hz
    
    @property
    def peak_fp32_flops_per_second(self) -> float:
        """峰值 FP32 性能 (FLOP/s)"""
        return self.peak_fp32_flops_per_cycle * self.sm_frequency_hz
    
    @property
    def achieved_dram_bandwidth_bytes_per_second(self) -> float:
        """实际达到的 DRAM 带宽 (bytes/s)"""
        if self.duration_ns <= 0:
            return 0
        return self.dram_bytes_total / (self.duration_ns * 1e-9)
    
    @property
    def peak_dram_bandwidth_bytes_per_second(self) -> float:
        """峰值 DRAM 带宽 (bytes/s) - 使用 DRAM 频率计算"""
        # 从 dram__bytes.sum.per_second 获取更准确的峰值
        # 这里用 peak_bytes_per_cycle * DRAM_frequency
        # 但 DRAM 频率可能不同于 SM 频率
        # 需要从 metrics 中获取实际的峰值带宽
        return self.dram_peak_bytes_per_cycle * self.sm_frequency_hz
    
    @property
    def arithmetic_intensity(self) -> float:
        """算术强度 (FLOP/Byte) for FP32 + DRAM"""
        if self.dram_bytes_total <= 0:
            return float('inf')
        # 计算总 FLOP 数
        total_cycles = self.sm_cycles
        total_flops = self.achieved_fp32_flops_per_cycle * total_cycles
        return total_flops / self.dram_bytes_total
    
    @property
    def ridge_point(self) -> float:
        """脊点 (Ridge Point) - 从 memory-bound 转换到 compute-bound 的 AI 值"""
        if self.peak_dram_bandwidth_bytes_per_second <= 0:
            return 0
        return self.peak_fp32_flops_per_second / self.peak_dram_bandwidth_bytes_per_second
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "kernel_name": self.kernel_name,
            "kernel_index": self.kernel_index,
            "duration_ns": self.duration_ns,
            "sm_cycles": self.sm_cycles,
            "sm_frequency_hz": self.sm_frequency_hz,
            "dram_bytes_read": self.dram_bytes_read,
            "dram_bytes_write": self.dram_bytes_write,
            "dram_bytes_total": self.dram_bytes_total,
            "dram_peak_bytes_per_cycle": self.dram_peak_bytes_per_cycle,
            "fp32_fadd_per_cycle": self.fp32_fadd_per_cycle,
            "fp32_fmul_per_cycle": self.fp32_fmul_per_cycle,
            "fp32_ffma_per_cycle": self.fp32_ffma_per_cycle,
            "fp32_peak_ffma_per_cycle": self.fp32_peak_ffma_per_cycle,
            "fp64_dadd_per_cycle": self.fp64_dadd_per_cycle,
            "fp64_dmul_per_cycle": self.fp64_dmul_per_cycle,
            "fp64_dfma_per_cycle": self.fp64_dfma_per_cycle,
            "fp64_peak_dfma_per_cycle": self.fp64_peak_dfma_per_cycle,
            # 计算值
            "achieved_fp32_flops_per_cycle": self.achieved_fp32_flops_per_cycle,
            "peak_fp32_flops_per_cycle": self.peak_fp32_flops_per_cycle,
            "achieved_fp32_flops_per_second": self.achieved_fp32_flops_per_second,
            "peak_fp32_flops_per_second": self.peak_fp32_flops_per_second,
            "achieved_dram_bandwidth_bytes_per_second": self.achieved_dram_bandwidth_bytes_per_second,
            "arithmetic_intensity_flop_per_byte": self.arithmetic_intensity,
            "ridge_point": self.ridge_point,
        }


class NcuReportParser:
    """NCU 报告解析器"""
    
    def __init__(self, report_path: str):
        self.report_path = report_path
        self._context = None
        self._device_info: Optional[DeviceInfo] = None
        self._roofline_data: List[RooflineData] = []
    
    def load(self) -> "NcuReportParser":
        """加载 NCU 报告"""
        import ncu_report
        self._context = ncu_report.load_report(self.report_path)
        return self
    
    def _get_metric_value(self, action, metric_name: str, default: float = 0.0) -> float:
        """安全获取 metric 值"""
        try:
            metric = action.metric_by_name(metric_name)
            if metric and metric.has_value():
                return float(metric.value())
        except Exception:
            pass
        return default
    
    def _get_metric_str(self, action, metric_name: str, default: str = "") -> str:
        """安全获取 metric 字符串值"""
        try:
            metric = action.metric_by_name(metric_name)
            if metric and metric.has_value():
                return str(metric.value())
        except Exception:
            pass
        return default
    
    def parse_device_info(self) -> DeviceInfo:
        """解析设备信息"""
        if self._context is None:
            raise RuntimeError("请先调用 load() 加载报告")
        
        action = self._context[0][0]  # 第一个 range 的第一个 action
        
        self._device_info = DeviceInfo(
            name=self._get_metric_str(action, "device__attribute_display_name").strip(),
            compute_capability=f"{self._get_metric_value(action, 'device__attribute_compute_capability_major'):.0f}.{self._get_metric_value(action, 'device__attribute_compute_capability_minor'):.0f}",
            sm_count=int(self._get_metric_value(action, "device__attribute_multiprocessor_count")),
            gpu_clock_rate_khz=int(self._get_metric_value(action, "device__attribute_clock_rate")),
            memory_clock_rate_khz=int(self._get_metric_value(action, "device__attribute_memory_clock_rate")),
            memory_bus_width=int(self._get_metric_value(action, "device__attribute_fb_bus_width")),
        )
        return self._device_info
    
    def parse_roofline_data(self) -> List[RooflineData]:
        """解析所有 kernel 的 Roofline 数据"""
        if self._context is None:
            raise RuntimeError("请先调用 load() 加载报告")
        
        self._roofline_data = []
        
        for range_obj in self._context:
            for idx, action in enumerate(range_obj):
                data = RooflineData(
                    kernel_name=action.name(),
                    kernel_index=idx,
                    
                    # 时间相关
                    duration_ns=self._get_metric_value(action, "gpu__time_duration.sum"),
                    sm_cycles=self._get_metric_value(action, "sm__cycles_elapsed.avg"),
                    sm_frequency_hz=self._get_metric_value(action, "sm__cycles_elapsed.avg.per_second"),
                    
                    # DRAM 相关
                    dram_bytes_read=self._get_metric_value(action, "dram__bytes_read.sum"),
                    dram_bytes_write=self._get_metric_value(action, "dram__bytes_write.sum"),
                    dram_bytes_total=self._get_metric_value(action, "dram__bytes_read.sum") + 
                                    self._get_metric_value(action, "dram__bytes_write.sum"),
                    dram_peak_bytes_per_cycle=self._get_metric_value(action, "dram__bytes.sum.peak_sustained"),
                    
                    # FP32 计算相关
                    fp32_fadd_per_cycle=self._get_metric_value(action, "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed"),
                    fp32_fmul_per_cycle=self._get_metric_value(action, "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed"),
                    fp32_ffma_per_cycle=self._get_metric_value(action, "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed"),
                    fp32_peak_ffma_per_cycle=self._get_metric_value(action, "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained"),
                    
                    # FP64 计算相关
                    fp64_dadd_per_cycle=self._get_metric_value(action, "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed"),
                    fp64_dmul_per_cycle=self._get_metric_value(action, "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed"),
                    fp64_dfma_per_cycle=self._get_metric_value(action, "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_cycle_elapsed"),
                    fp64_peak_dfma_per_cycle=self._get_metric_value(action, "sm__sass_thread_inst_executed_op_dfma_pred_on.sum.peak_sustained"),
                )
                self._roofline_data.append(data)
        
        return self._roofline_data
    
    def parse_all_metrics(self, action_index: int = 0) -> Dict[str, Any]:
        """解析指定 action 的所有 metrics (用于调试)"""
        if self._context is None:
            raise RuntimeError("请先调用 load() 加载报告")
        
        action = self._context[0][action_index]
        metrics = {}
        
        for name in action.metric_names():
            metric = action.metric_by_name(name)
            if metric and metric.has_value():
                try:
                    val = metric.value()
                    unit = metric.unit() if metric.unit() else ""
                    metrics[name] = {"value": val, "unit": unit}
                except Exception:
                    pass
        
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """获取报告摘要"""
        if self._device_info is None:
            self.parse_device_info()
        if not self._roofline_data:
            self.parse_roofline_data()
        
        return {
            "report_path": self.report_path,
            "device_info": {
                "name": self._device_info.name,
                "compute_capability": self._device_info.compute_capability,
                "sm_count": self._device_info.sm_count,
                "gpu_clock_rate_mhz": self._device_info.gpu_clock_rate_khz / 1000,
                "memory_clock_rate_mhz": self._device_info.memory_clock_rate_khz / 1000,
                "memory_bus_width_bits": self._device_info.memory_bus_width,
            },
            "kernel_count": len(self._roofline_data),
            "kernels": [d.to_dict() for d in self._roofline_data],
        }
    
    def export_json(self, output_path: str) -> None:
        """导出 JSON 格式"""
        summary = self.get_summary()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="解析 NCU .ncu-rep 报告文件，提取 Roofline 分析数据"
    )
    parser.add_argument("--report", "-r", required=True, help="输入 .ncu-rep 文件路径")
    parser.add_argument("--output", "-o", help="输出 JSON 文件路径")
    parser.add_argument("--dump-metrics", action="store_true", help="导出所有 metrics (调试用)")
    parser.add_argument("--kernel-index", type=int, default=0, help="指定 kernel 索引 (用于 --dump-metrics)")
    
    args = parser.parse_args()
    
    print(f"加载报告: {args.report}")
    parser_obj = NcuReportParser(args.report).load()
    
    if args.dump_metrics:
        metrics = parser_obj.parse_all_metrics(args.kernel_index)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"Metrics 已导出到: {args.output}")
        else:
            for name, data in sorted(metrics.items()):
                print(f"{name}: {data['value']} {data['unit']}")
        return
    
    summary = parser_obj.get_summary()
    
    # 打印摘要
    print(f"\n===== 设备信息 =====")
    dev = summary["device_info"]
    print(f"  设备名称: {dev['name']}")
    print(f"  计算能力: {dev['compute_capability']}")
    print(f"  SM 数量: {dev['sm_count']}")
    print(f"  GPU 时钟: {dev['gpu_clock_rate_mhz']:.0f} MHz")
    print(f"  显存时钟: {dev['memory_clock_rate_mhz']:.0f} MHz")
    print(f"  显存位宽: {dev['memory_bus_width_bits']} bits")
    
    print(f"\n===== Kernel 数量: {summary['kernel_count']} =====")
    
    for i, k in enumerate(summary["kernels"]):
        print(f"\n--- Kernel {i}: {k['kernel_name']} ---")
        print(f"  执行时间: {k['duration_ns'] / 1000:.2f} µs")
        print(f"  SM 频率: {k['sm_frequency_hz'] / 1e9:.3f} GHz")
        print(f"  DRAM 读取: {k['dram_bytes_read'] / 1024 / 1024:.2f} MB")
        print(f"  DRAM 写入: {k['dram_bytes_write'] / 1024 / 1024:.2f} MB")
        print(f"  FP32 实际性能: {k['achieved_fp32_flops_per_second'] / 1e9:.2f} GFLOP/s")
        print(f"  FP32 峰值性能: {k['peak_fp32_flops_per_second'] / 1e12:.2f} TFLOP/s")
        print(f"  算术强度 (AI): {k['arithmetic_intensity_flop_per_byte']:.4f} FLOP/Byte")
    
    if args.output:
        parser_obj.export_json(args.output)
        print(f"\n完整数据已导出到: {args.output}")


if __name__ == "__main__":
    main()
