#!/usr/bin/env python3
"""
Roofline Plot - 基于 NCU Parser 的 Roofline 绘图工具

根据 ncu_parser.py 解析的数据绘制标准 Roofline 图。
支持 DRAM + FP32 场景的 Roofline 分析。

使用方法：
    python plot_roofline_api.py --report <ncu-rep文件> --output <输出图片>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ncu_parser import NcuReportParser, RooflineData


def format_flops(flops: float) -> str:
    """格式化 FLOP/s 为人类可读格式"""
    if flops >= 1e15:
        return f"{flops / 1e15:.2f} PFLOP/s"
    elif flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOP/s"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOP/s"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOP/s"
    else:
        return f"{flops:.2f} FLOP/s"


def format_bandwidth(bw: float) -> str:
    """格式化带宽为人类可读格式"""
    if bw >= 1e12:
        return f"{bw / 1e12:.2f} TB/s"
    elif bw >= 1e9:
        return f"{bw / 1e9:.2f} GB/s"
    elif bw >= 1e6:
        return f"{bw / 1e6:.2f} MB/s"
    else:
        return f"{bw:.2f} B/s"


class RooflinePlotter:
    """Roofline 绘图器"""
    
    def __init__(
        self,
        peak_fp32_flops: float,  # FLOP/s
        peak_dram_bandwidth: float,  # Bytes/s
        device_name: str = "",
    ):
        self.peak_fp32_flops = peak_fp32_flops
        self.peak_dram_bandwidth = peak_dram_bandwidth
        self.device_name = device_name
        
        # 计算脊点 (Ridge Point)
        self.ridge_point = peak_fp32_flops / peak_dram_bandwidth
        
        # 数据点
        self.points: List[Tuple[float, float, str]] = []  # (AI, FLOP/s, label)
    
    def add_point(self, ai: float, perf: float, label: str = ""):
        """添加一个数据点"""
        self.points.append((ai, perf, label))
    
    def plot(
        self,
        output_path: str,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 150,
        show_labels: bool = False,
        annotate_roofs: bool = True,
        show_ridge_line: bool = True,
    ):
        """绑制 Roofline 图"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # 设置对数坐标轴
        ax.set_xscale("log")
        ax.set_yscale("log")
        
        # 确定坐标范围
        if self.points:
            ais = [p[0] for p in self.points]
            perfs = [p[1] for p in self.points]
            ai_min = min(ais) / 10
            ai_max = max(ais) * 10
            perf_min = min(perfs) / 10
            perf_max = max(max(perfs), self.peak_fp32_flops) * 2
        else:
            ai_min = 0.01
            ai_max = 1000
            perf_min = 1e9
            perf_max = self.peak_fp32_flops * 2
        
        # 确保包含脊点
        ai_min = min(ai_min, self.ridge_point / 10)
        ai_max = max(ai_max, self.ridge_point * 10)
        
        # 生成 x 轴数据点
        x = np.logspace(np.log10(ai_min), np.log10(ai_max), 500)
        
        # 绘制 Roofline
        # Memory-bound 区域: Performance = BW × AI
        # Compute-bound 区域: Performance = Peak
        y_memory_bound = self.peak_dram_bandwidth * x
        y_compute_bound = np.full_like(x, self.peak_fp32_flops)
        y_roofline = np.minimum(y_memory_bound, y_compute_bound)
        
        # 绘制屋顶线
        ax.plot(x, y_roofline, 'b-', linewidth=2.5, label='Roofline', zorder=5)
        
        # 单独绘制两个区域的线条用于标注
        # Memory-bound 斜线
        x_mem = x[x <= self.ridge_point * 1.1]
        y_mem = self.peak_dram_bandwidth * x_mem
        ax.plot(x_mem, y_mem, 'b-', linewidth=2.5, zorder=5)
        
        # Compute-bound 水平线
        x_comp = x[x >= self.ridge_point * 0.9]
        y_comp = np.full_like(x_comp, self.peak_fp32_flops)
        ax.plot(x_comp, y_comp, 'b-', linewidth=2.5, zorder=5)
        
        # 绘制脊点标记
        if show_ridge_line:
            ax.axvline(x=self.ridge_point, color='gray', linestyle='--', 
                      linewidth=1, alpha=0.7, zorder=3)
            ax.plot(self.ridge_point, self.peak_fp32_flops, 'b^', 
                   markersize=10, zorder=6, label=f'Ridge Point (AI={self.ridge_point:.2f})')
        
        # 标注峰值
        if annotate_roofs:
            # 标注峰值性能
            ax.annotate(
                f'Peak FP32: {format_flops(self.peak_fp32_flops)}',
                xy=(ai_max * 0.5, self.peak_fp32_flops),
                xytext=(ai_max * 0.3, self.peak_fp32_flops * 1.3),
                fontsize=10,
                ha='center',
                arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
            )
            
            # 标注峰值带宽（在斜线上）
            bw_label_ai = self.ridge_point / 5
            bw_label_perf = self.peak_dram_bandwidth * bw_label_ai
            ax.annotate(
                f'Peak DRAM BW: {format_bandwidth(self.peak_dram_bandwidth)}',
                xy=(bw_label_ai, bw_label_perf),
                xytext=(bw_label_ai * 1.5, bw_label_perf * 0.3),
                fontsize=10,
                ha='left',
                arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
            )
        
        # 绘制数据点
        if self.points:
            ais = [p[0] for p in self.points]
            perfs = [p[1] for p in self.points]
            labels = [p[2] for p in self.points]
            
            scatter = ax.scatter(ais, perfs, c='red', s=80, alpha=0.8, 
                                zorder=10, label='Kernel Data Points', edgecolors='darkred')
            
            # 添加标签
            if show_labels:
                for ai, perf, label in self.points:
                    ax.annotate(
                        label[:30] if len(label) > 30 else label,
                        (ai, perf),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=7,
                        alpha=0.8,
                    )
        
        # 设置坐标范围
        ax.set_xlim(ai_min, ai_max)
        ax.set_ylim(perf_min, perf_max)
        
        # 设置标签和标题
        ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=12)
        ax.set_ylabel('Performance (FLOP/s)', fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14)
        elif self.device_name:
            ax.set_title(f'Roofline Analysis - {self.device_name}', fontsize=14)
        else:
            ax.set_title('Roofline Analysis (DRAM + FP32)', fontsize=14)
        
        # 添加网格
        ax.grid(True, which='both', linestyle=':', alpha=0.5)
        
        # 添加图例
        ax.legend(loc='lower right', fontsize=10)
        
        # 添加说明文字
        info_text = (
            f"Peak FP32: {format_flops(self.peak_fp32_flops)}\n"
            f"Peak DRAM BW: {format_bandwidth(self.peak_dram_bandwidth)}\n"
            f"Ridge Point: {self.ridge_point:.2f} FLOP/Byte"
        )
        ax.text(
            0.02, 0.98, info_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        )
        
        # 保存图片
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        return output_path


def calculate_theoretical_peak_bandwidth(
    memory_clock_khz: int,
    bus_width_bits: int,
) -> float:
    """
    计算理论峰值 DRAM 带宽
    
    Args:
        memory_clock_khz: 显存时钟频率 (kHz)
        bus_width_bits: 显存总线宽度 (bits)
    
    Returns:
        峰值带宽 (bytes/s)
    """
    # HBM 使用 DDR，所以乘以 2
    memory_clock_hz = memory_clock_khz * 1000
    bus_width_bytes = bus_width_bits // 8
    return memory_clock_hz * bus_width_bytes * 2


def main():
    parser = argparse.ArgumentParser(
        description="基于 NCU 报告绘制 Roofline 图 (DRAM + FP32)"
    )
    parser.add_argument("--report", "-r", required=True, help="输入 .ncu-rep 文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出图片路径 (.png/.pdf)")
    parser.add_argument("--title", "-t", help="图标题")
    parser.add_argument("--show-labels", action="store_true", help="显示 kernel 名称标签")
    parser.add_argument("--json-output", help="导出 JSON 数据文件")
    parser.add_argument("--kernel-filter", help="kernel 名称过滤（正则表达式）")
    parser.add_argument("--use-theoretical-bw", action="store_true", 
                       help="使用理论峰值带宽而非测量值")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")
    
    args = parser.parse_args()
    
    # 加载和解析报告
    print(f"加载报告: {args.report}")
    ncu_parser = NcuReportParser(args.report).load()
    
    device_info = ncu_parser.parse_device_info()
    roofline_data = ncu_parser.parse_roofline_data()
    
    print(f"\n设备: {device_info.name}")
    print(f"计算能力: {device_info.compute_capability}")
    print(f"SM 数量: {device_info.sm_count}")
    
    # 过滤 kernel
    if args.kernel_filter:
        import re
        pattern = re.compile(args.kernel_filter)
        roofline_data = [d for d in roofline_data if pattern.search(d.kernel_name)]
        print(f"过滤后 kernel 数量: {len(roofline_data)}")
    
    if not roofline_data:
        print("错误: 没有找到有效的 kernel 数据")
        sys.exit(1)
    
    # 计算峰值性能（取所有 kernel 中的最大值作为参考）
    peak_fp32_flops = max(d.peak_fp32_flops_per_second for d in roofline_data)
    
    # 计算峰值带宽
    if args.use_theoretical_bw:
        peak_dram_bw = calculate_theoretical_peak_bandwidth(
            device_info.memory_clock_rate_khz,
            device_info.memory_bus_width,
        )
        print(f"使用理论峰值带宽: {format_bandwidth(peak_dram_bw)}")
    else:
        # 从 kernel 数据中获取
        # 使用 dram_peak_bytes_per_cycle * SM 频率作为峰值带宽估计
        # 但更准确的是使用实际测量的 DRAM 吞吐量与利用率
        avg_sm_freq = sum(d.sm_frequency_hz for d in roofline_data) / len(roofline_data)
        peak_bytes_per_cycle = max(d.dram_peak_bytes_per_cycle for d in roofline_data)
        peak_dram_bw = peak_bytes_per_cycle * avg_sm_freq
        
        # 或者直接用理论值作为更可靠的峰值
        theoretical_bw = calculate_theoretical_peak_bandwidth(
            device_info.memory_clock_rate_khz,
            device_info.memory_bus_width,
        )
        # 使用较大的值
        peak_dram_bw = max(peak_dram_bw, theoretical_bw)
        print(f"峰值带宽: {format_bandwidth(peak_dram_bw)} (理论值)")
    
    print(f"峰值 FP32 性能: {format_flops(peak_fp32_flops)}")
    
    # 创建绘图器
    plotter = RooflinePlotter(
        peak_fp32_flops=peak_fp32_flops,
        peak_dram_bandwidth=peak_dram_bw,
        device_name=device_info.name,
    )
    
    # 添加数据点
    print(f"\n===== Roofline 数据点 =====")
    print(f"{'Kernel':<40} {'AI (FLOP/B)':<15} {'Perf (GFLOP/s)':<18} {'% of Peak':<10}")
    print("-" * 90)
    
    for data in roofline_data:
        ai = data.arithmetic_intensity
        perf = data.achieved_fp32_flops_per_second
        pct_peak = (perf / peak_fp32_flops) * 100
        
        if args.verbose:
            print(f"{data.kernel_name[:40]:<40} {ai:<15.4f} {perf/1e9:<18.2f} {pct_peak:<10.2f}%")
        
        plotter.add_point(ai, perf, data.kernel_name)
    
    # 打印摘要
    avg_ai = sum(d.arithmetic_intensity for d in roofline_data) / len(roofline_data)
    avg_perf = sum(d.achieved_fp32_flops_per_second for d in roofline_data) / len(roofline_data)
    
    print("-" * 90)
    print(f"{'平均值':<40} {avg_ai:<15.4f} {avg_perf/1e9:<18.2f}")
    
    ridge_point = peak_fp32_flops / peak_dram_bw
    print(f"\n脊点 (Ridge Point): {ridge_point:.4f} FLOP/Byte")
    
    if avg_ai < ridge_point:
        print("状态: Memory-bound (内存受限)")
    else:
        print("状态: Compute-bound (计算受限)")
    
    # 绘制图表
    print(f"\n绑制 Roofline 图...")
    plotter.plot(
        output_path=args.output,
        title=args.title,
        show_labels=args.show_labels,
    )
    print(f"图表已保存到: {args.output}")
    
    # 导出 JSON
    if args.json_output:
        summary = ncu_parser.get_summary()
        summary["roofline_analysis"] = {
            "peak_fp32_flops_per_second": peak_fp32_flops,
            "peak_dram_bandwidth_bytes_per_second": peak_dram_bw,
            "ridge_point_flop_per_byte": ridge_point,
            "avg_arithmetic_intensity": avg_ai,
            "avg_achieved_fp32_flops_per_second": avg_perf,
        }
        with open(args.json_output, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"JSON 数据已导出到: {args.json_output}")
    
    # 打印验证信息
    print("\n" + "=" * 60)
    print("===== 用于与 NCU GUI 对比验证的关键数据 =====")
    print("=" * 60)
    print(f"\n以下数据可与 Nsight Compute GUI 中的 'Speed of Light' 和")
    print(f"'GPU Speed Of Light Roofline Chart' section 对比验证:\n")
    
    sample = roofline_data[0]
    print(f"Kernel: {sample.kernel_name}")
    print(f"  - 执行时间 (gpu__time_duration.sum): {sample.duration_ns:.2f} ns")
    print(f"  - SM 周期 (sm__cycles_elapsed.avg): {sample.sm_cycles:.2f} cycles")
    print(f"  - SM 频率 (sm__cycles_elapsed.avg.per_second): {sample.sm_frequency_hz/1e9:.3f} GHz")
    print(f"  - DRAM 读取 (dram__bytes_read.sum): {sample.dram_bytes_read/1024/1024:.2f} MB")
    print(f"  - DRAM 写入 (dram__bytes_write.sum): {sample.dram_bytes_write/1024/1024:.2f} MB")
    print(f"  - DRAM 峰值 (dram__bytes.sum.peak_sustained): {sample.dram_peak_bytes_per_cycle:.0f} byte/cycle")
    print(f"  - FP32 FFMA/cycle (smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed): {sample.fp32_ffma_per_cycle:.2f}")
    print(f"  - FP32 FADD/cycle (smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed): {sample.fp32_fadd_per_cycle:.2f}")
    print(f"  - FP32 FMUL/cycle (smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed): {sample.fp32_fmul_per_cycle:.2f}")
    print(f"  - FP32 峰值 FFMA/cycle (sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained): {sample.fp32_peak_ffma_per_cycle:.0f}")
    
    print(f"\n计算公式验证:")
    print(f"  - 实际 FP32 FLOP/cycle = FADD + FMUL + 2*FFMA")
    print(f"    = {sample.fp32_fadd_per_cycle:.2f} + {sample.fp32_fmul_per_cycle:.2f} + 2*{sample.fp32_ffma_per_cycle:.2f}")
    print(f"    = {sample.achieved_fp32_flops_per_cycle:.2f} FLOP/cycle")
    print(f"  - 实际 FP32 性能 = FLOP/cycle × SM频率")
    print(f"    = {sample.achieved_fp32_flops_per_cycle:.2f} × {sample.sm_frequency_hz/1e9:.3f} GHz")
    print(f"    = {sample.achieved_fp32_flops_per_second/1e9:.2f} GFLOP/s")
    print(f"  - 峰值 FP32 FLOP/cycle = 2 × peak_FFMA/cycle")
    print(f"    = 2 × {sample.fp32_peak_ffma_per_cycle:.0f} = {sample.peak_fp32_flops_per_cycle:.0f} FLOP/cycle")
    print(f"  - 峰值 FP32 性能 = {sample.peak_fp32_flops_per_cycle:.0f} × {sample.sm_frequency_hz/1e9:.3f} GHz")
    print(f"    = {sample.peak_fp32_flops_per_second/1e12:.2f} TFLOP/s")
    print(f"  - 算术强度 = 总FLOPs / 总DRAM流量")
    total_flops = sample.achieved_fp32_flops_per_cycle * sample.sm_cycles
    print(f"    = ({sample.achieved_fp32_flops_per_cycle:.2f} × {sample.sm_cycles:.0f}) / {sample.dram_bytes_total/1e6:.2f}M")
    print(f"    = {total_flops/1e6:.2f}M / {sample.dram_bytes_total/1e6:.2f}M")
    print(f"    = {sample.arithmetic_intensity:.4f} FLOP/Byte")


if __name__ == "__main__":
    main()
