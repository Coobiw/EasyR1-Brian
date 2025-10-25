#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制 Reject Sampling 结果的 Box Plot
横坐标为 epoch，纵坐标为 min_error
"""

import os
import json
import re
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

# 设置字体（使用系统默认字体，避免字体找不到的问题）
# matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

def extract_epoch_from_model_name(model_name, steps_per_epoch=37):
    """
    从模型名称中提取 epoch 数
    
    Args:
        model_name: 模型名称，如 "qwen_2_5_vl_mergedata_step185"
        steps_per_epoch: 每个 epoch 的 step 数
    
    Returns:
        epoch 数（float）
    """
    # 如果是 baseline，返回 0
    if 'baseline' in model_name.lower():
        return 0.0
    
    # 提取 step 数字
    match = re.search(r'step[_-]?(\d+)', model_name, re.IGNORECASE)
    if match:
        step_num = int(match.group(1))
        epoch = step_num / steps_per_epoch
        return epoch
    
    # 如果没找到 step，尝试提取 epoch
    match = re.search(r'epoch[_-]?(\d+)', model_name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    
    # 默认返回 None
    return None

def load_results(result_dir):
    """
    加载单个模型的 reject sampling 结果
    
    Args:
        result_dir: 结果目录路径
    
    Returns:
        (model_name, epoch, errors): 元组，包含模型名称、epoch数和误差列表
    """
    result_file = Path(result_dir) / "reject_sampling_results.json"
    
    if not result_file.exists():
        return None
    
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        # 提取所有有效的 min_error
        errors = [item['min_error'] for item in data if item['min_error'] != float('inf')]
        
        if not errors:
            return None
        
        # 从目录名获取模型名称
        model_name = Path(result_dir).name
        
        # 提取 epoch
        epoch = extract_epoch_from_model_name(model_name)
        
        if epoch is None:
            print(f"Warning: Could not extract epoch from {model_name}, skipping...")
            return None
        
        return (model_name, epoch, errors)
    
    except Exception as e:
        print(f"Error loading {result_file}: {e}")
        return None

def plot_boxplot(results_data, output_file="reject_sampling_boxplot.png", 
                 steps_per_epoch=37, figsize=(16, 8)):
    """
    绘制 box plot (科研风格)
    
    Args:
        results_data: list of (model_name, epoch, errors)
        output_file: 输出文件名
        steps_per_epoch: 每个 epoch 的 step 数
        figsize: 图片大小
    """
    # 按 epoch 排序
    results_data.sort(key=lambda x: x[1])
    
    # 过滤掉 epoch≈4.9 的数据（best_ckpt, step 181）
    # step 181/37 ≈ 4.89，过滤掉 [4.8, 5.0) 区间的数据
    results_data = [item for item in results_data if not (4.8 <= item[1] < 5.0)]
    
    # 准备数据
    epochs = [item[1] for item in results_data]
    model_names = [item[0] for item in results_data]
    errors_list = [item[2] for item in results_data]
    
    # 计算均值和标准差
    means = [np.mean(errors) for errors in errors_list]
    stds = [np.std(errors) for errors in errors_list]
    
    # 找到 baseline (epoch 0) 的均值和标准差
    baseline_idx = epochs.index(0.0) if 0.0 in epochs else None
    if baseline_idx is not None:
        baseline_mean = means[baseline_idx]
        baseline_std = stds[baseline_idx]
    else:
        baseline_mean = None
        baseline_std = None
    
    # 创建图表 - 使用 GridSpec 来布局
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 2, width_ratios=[5, 1], wspace=0.0)
    
    # 主图（左侧）
    ax = fig.add_subplot(gs[0])
    # 右侧分布曲线图
    ax_hist = fig.add_subplot(gs[1], sharey=ax)
    
    # 设置背景颜色（浅灰色）
    ax.set_facecolor('#f8f8f8')
    
    # 如果有 baseline，绘制误差范围区域（垂直区域，平行于 y 轴）
    if baseline_mean is not None:
        ax.axhspan(baseline_mean - baseline_std, baseline_mean + baseline_std,
                   alpha=0.15, color='gray', zorder=0,
                   label=f'Baseline range (mean±std)')
        ax.axhline(baseline_mean, color='gray', linestyle='--', linewidth=1.5, 
                   alpha=0.5, zorder=1, label='Baseline mean')
    
    # 绘制 box plot（不显示均值标记，我们稍后单独画）
    bp = ax.boxplot(errors_list, 
                     positions=epochs,
                     widths=0.25,
                     patch_artist=True,
                     showmeans=False,
                     showfliers=False,  # 不显示异常值
                     medianprops=dict(color='darkblue', linewidth=2))
    
    # 设置 box 颜色 - 使用更科研的配色
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(results_data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    
    # 在 box 上叠加散点（泡泡形状）- 使用更鲜艳的颜色
    np.random.seed(42)  # 保证可重复性
    # 使用科研风格的鲜艳颜色系列 - 最后一个改为亮青色，更容易看清
    scatter_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf']
    for i, (epoch, errors) in enumerate(zip(epochs, errors_list)):
        # 随机采样部分数据点，避免太密集
        sample_size = min(len(errors), 200)
        sampled_errors = np.random.choice(errors, sample_size, replace=False)
        
        # 添加一些随机抖动，使点分散开
        x_jitter = np.random.normal(epoch, 0.03, sample_size)
        
        # 绘制散点（泡泡）- 使用更鲜艳的颜色
        scatter_color = scatter_colors[i % len(scatter_colors)]
        ax.scatter(x_jitter, sampled_errors, 
                  alpha=0.4, s=25, 
                  color=scatter_color, 
                  edgecolors='white', 
                  linewidths=0.8,
                  zorder=3)
    
    # 绘制连接均值的线
    ax.plot(epochs, means, 
            color='red', 
            linewidth=3, 
            marker='o', 
            markersize=10,
            markerfacecolor='red',
            markeredgecolor='darkred',
            markeredgewidth=2,
            label='Mean error',
            zorder=10,
            linestyle='-',
            alpha=0.9)
    
    # 在均值点上添加数值标签
    for epoch, mean in zip(epochs, means):
        ax.annotate(f'{mean:.3f}', 
                   xy=(epoch, mean), 
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=9,
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   zorder=11)
    
    # 设置坐标轴范围和标签
    ax.set_ylim(0, 2)
    ax.set_xlabel('Epoch', fontsize=15, fontweight='bold')
    ax.set_ylabel('Minimum Error', fontsize=15, fontweight='bold')
    ax.set_title('Reject Sampling Error Distribution across Training', 
                 fontsize=18, fontweight='bold', pad=20)
    
    # 设置 x 轴刻度
    ax.set_xticks(epochs)
    ax.set_xticklabels([f'{e:.1f}' for e in epochs], fontsize=12)
    
    # 设置 y 轴刻度
    ax.set_yticks(np.arange(0, 2.5, 0.5))
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', axis='y', zorder=0)
    ax.set_axisbelow(True)
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # 添加统计信息文本 - 左上角，只显示baseline信息
    if baseline_mean is not None:
        stats_text = f"Baseline: {baseline_mean:.4f} ± {baseline_std:.4f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 绘制右侧分布曲线（垂直方向）
    # 计算直方图数据 (bins: [0, 0.5), [0.5, 1.0), [1.0, 1.5), [1.5, 2.0))
    from scipy.interpolate import make_interp_spline
    
    bins = np.arange(0, 2.5, 0.5)
    bin_centers = bins[:-1] + 0.25  # 每个柱子的中心位置
    
    # 为每个 epoch 绘制平滑曲线，显示分布差异
    for i, (epoch, errors) in enumerate(zip(epochs, errors_list)):
        color = scatter_colors[i % len(scatter_colors)]
        
        # 计算直方图
        hist, _ = np.histogram(errors, bins=bins)
        # 归一化到 [0, 1]
        hist_norm = hist / hist.max() if hist.max() > 0 else hist
        
        # 使用样条插值创建平滑曲线
        # 添加边界点使曲线更自然
        y_points = np.concatenate([[0], bin_centers, [2.0]])
        x_points = np.concatenate([[0], hist_norm * 0.8, [0]])
        
        # 创建更密集的点用于绘制平滑曲线
        y_smooth = np.linspace(0, 2.0, 100)
        
        # 使用三次样条插值
        if len(y_points) > 3:  # 需要至少4个点
            spl = make_interp_spline(y_points, x_points, k=3)
            x_smooth = spl(y_smooth)
            # 绘制垂直曲线（向右延伸）
            ax_hist.plot(x_smooth, y_smooth, color=color, linewidth=2.5, 
                        alpha=0.7, linestyle='-', zorder=10)
    
    # 设置右侧分布曲线图的样式 - 共享Y轴，不显示额外标签
    ax_hist.set_ylim(0, 2)
    # X轴范围：向右延伸，显示完整曲线
    ax_hist.set_xlim(-0.05, 1.1)
    # 隐藏Y轴刻度标签（共享主图的Y轴）
    ax_hist.set_yticklabels([])
    ax_hist.set_xticks([])  # 不显示X轴刻度
    ax_hist.tick_params(left=False, right=False)  # 隐藏Y轴刻度线
    ax_hist.grid(False)  # 不显示网格
    # 设置背景颜色与主图一致
    # ax_hist.set_facecolor('#f8f8f8')
    # 隐藏所有边框
    ax_hist.spines['right'].set_visible(False)
    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['bottom'].set_visible(False)
    ax_hist.spines['left'].set_visible(False)
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Box plot saved to: {output_file}")
    
    # 显示图片
    # plt.show()
    plt.close()

def print_summary(results_data):
    """打印统计摘要"""
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTotal checkpoints: {len(results_data)}")
    print("\nDetailed statistics:")
    print(f"{'Epoch':<8} {'Model Name':<40} {'Samples':<10} {'Mean':<10} {'Median':<10}")
    print("-" * 80)
    
    for model_name, epoch, errors in sorted(results_data, key=lambda x: x[1]):
        mean_err = np.mean(errors)
        median_err = np.median(errors)
        short_name = model_name[:38] + ".." if len(model_name) > 40 else model_name
        print(f"{epoch:<8.1f} {short_name:<40} {len(errors):<10} {mean_err:<10.4f} {median_err:<10.4f}")
    
    print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description="Plot box plot for reject sampling results")
    parser.add_argument("--results_dir", type=str, default="./reject_sample_results",
                        help="Directory containing reject sampling results")
    parser.add_argument("--output", type=str, default="reject_sampling_boxplot.png",
                        help="Output file path")
    parser.add_argument("--steps_per_epoch", type=int, default=37,
                        help="Number of steps per epoch")
    parser.add_argument("--figsize", type=str, default="16,8",
                        help="Figure size (width,height)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed statistics")
    args = parser.parse_args()
    
    # 解析 figsize
    figsize = tuple(map(float, args.figsize.split(',')))
    
    print("="*80)
    print("Reject Sampling Box Plot Generator")
    print("="*80)
    print(f"\nResults directory: {args.results_dir}")
    print(f"Steps per epoch: {args.steps_per_epoch}")
    print(f"Output file: {args.output}")
    print()
    
    # 扫描所有结果目录
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"❌ Error: Results directory not found: {results_dir}")
        return
    
    print("Scanning for results...")
    results_data = []
    
    for subdir in results_dir.iterdir():
        if subdir.is_dir():
            result = load_results(subdir)
            if result:
                model_name, epoch, errors = result
                print(f"  ✅ Loaded: {model_name} (epoch {epoch:.1f}, {len(errors)} samples)")
                results_data.append(result)
            else:
                if args.verbose:
                    print(f"  ⚠️  Skipped: {subdir.name}")
    
    if not results_data:
        print("\n❌ No valid results found!")
        return
    
    print(f"\n✅ Successfully loaded {len(results_data)} checkpoints")
    
    # 打印统计摘要
    if args.verbose:
        print_summary(results_data)
    
    # 绘制 box plot
    print("\nGenerating box plot...")
    plot_boxplot(results_data, args.output, args.steps_per_epoch, figsize)
    
    # 保存详细的统计信息到文本文件
    stats_file = Path(args.output).parent / "boxplot_statistics.txt"
    with open(stats_file, 'w') as f:
        f.write("Reject Sampling Statistics\n")
        f.write("="*80 + "\n\n")
        f.write(f"Steps per epoch: {args.steps_per_epoch}\n")
        f.write(f"Total checkpoints: {len(results_data)}\n\n")
        f.write(f"{'Epoch':<8} {'Model Name':<40} {'Samples':<10} {'Mean':<10} {'Median':<10} {'Std':<10}\n")
        f.write("-" * 90 + "\n")
        
        # 定义分布区间 (bins)
        dist_bins = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, float('inf')]
        dist_labels = ['[0, 0.25)', '[0.25, 0.5)', '[0.5, 0.75)', '[0.75, 1.0)', 
                      '[1.0, 1.5)', '[1.5, 2.0)', '[2.0, inf)']
        
        for model_name, epoch, errors in sorted(results_data, key=lambda x: x[1]):
            mean_err = np.mean(errors)
            median_err = np.median(errors)
            std_err = np.std(errors)
            f.write(f"{epoch:<8.1f} {model_name:<40} {len(errors):<10} {mean_err:<10.4f} {median_err:<10.4f} {std_err:<10.4f}\n")
            
            # 计算误差分布
            hist, _ = np.histogram(errors, bins=dist_bins)
            f.write(f"  Error Distribution:\n")
            for i, (label, count) in enumerate(zip(dist_labels, hist)):
                percentage = (count / len(errors)) * 100
                f.write(f"    {label:<15} {count:>5} ({percentage:>5.2f}%)\n")
            f.write("\n")
    
    print(f"✅ Statistics saved to: {stats_file}")
    print("\n" + "="*80)
    print("Done!")
    print("="*80)

if __name__ == "__main__":
    main()

