#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from pathlib import Path
import argparse
import logging
import warnings
try:
    from utils.logging_utils import setup_logging
except ImportError:
    import os as _os
    import sys as _sys

    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from utils.logging_utils import setup_logging
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

try:
    from .paths import collect_run_dirs, filter_by_sweep_timestamp, pick_latest_sweep_timestamp
    from .summary import summarize_runs
except ImportError:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from paths import collect_run_dirs, filter_by_sweep_timestamp, pick_latest_sweep_timestamp
    from summary import summarize_runs

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


def calculate_malicious_ratio(df):
    total_users = df['Normal Users'] + df['Malicious Users']
    df['Malicious User Ratio (%)'] = (df['Malicious Users'] / total_users * 100).fillna(0)
    df['Malicious User Ratio Standardized'] = df['Malicious User Ratio (%)'].apply(
        lambda x: 0 if x == 0 else (25 if x <= 25 else (50 if x <= 50 else (75 if x <= 75 else 100)))
    )
    return df


def plot_malicious_ratio_vs_performance(df, output_dir):
    logger.info("=" * 60)
    logger.info("Malicious user ratio vs overall system performance degradation")
    logger.info("=" * 60)
    
    df = calculate_malicious_ratio(df)
    
    attack_methods = df['Attack Method'].unique()
    attack_methods = [m for m in attack_methods if pd.notna(m) and m != '']
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('System Performance Degradation vs Malicious User Ratio', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    normal_metrics = [
        ('Normal User Goodput Rate', 'Normal User Goodput Rate', axes[0, 0]),
        ('Normal User Average Latency (s)', 'Normal User Average Latency', axes[0, 1]),
        ('Normal User Average TTFT (s)', 'Normal User TTFT', axes[0, 2]),
        ('Normal User Average TPOT (s)', 'Normal User TPOT', axes[0, 3]),
    ]
    
    malicious_metrics = [
        ('Malicious User Goodput Rate', 'Malicious User Goodput Rate', axes[1, 0]),
        ('Malicious User Average Latency (s)', 'Malicious User Average Latency', axes[1, 1]),
        ('Malicious User Average TTFT (s)', 'Malicious User TTFT', axes[1, 2]),
        ('Malicious User Average TPOT (s)', 'Malicious User TPOT', axes[1, 3]),
    ]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(attack_methods)))
    color_map = {method: colors[i] for i, method in enumerate(attack_methods)}
    
    for metric_col, metric_name, ax in normal_metrics:
        for method in attack_methods:
            method_data = df[df['Attack Method'] == method].copy()
            if len(method_data) == 0:
                continue
            
            method_data = method_data[method_data['Malicious User Ratio Standardized'] < 100].copy()
            if len(method_data) == 0:
                continue
            
            grouped = method_data.groupby('Malicious User Ratio Standardized')[metric_col].mean().reset_index()
            grouped = grouped.sort_values('Malicious User Ratio Standardized')
            
            if len(grouped) > 0:
                ax.plot(grouped['Malicious User Ratio Standardized'], grouped[metric_col],
                       marker='o', linewidth=2, label=method, 
                       color=color_map[method], markersize=6, alpha=0.8)
        
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Malicious User Ratio (%)', fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8, ncol=2)
    
    for metric_col, metric_name, ax in malicious_metrics:
        for method in attack_methods:
            method_data = df[df['Attack Method'] == method].copy()
            if len(method_data) == 0:
                continue
            
            method_data = method_data[method_data['Malicious User Ratio Standardized'] > 0].copy()
            if len(method_data) == 0:
                continue
            
            grouped = method_data.groupby('Malicious User Ratio Standardized')[metric_col].mean().reset_index()
            grouped = grouped.sort_values('Malicious User Ratio Standardized')
            
            if len(grouped) > 0:
                ax.plot(grouped['Malicious User Ratio Standardized'], grouped[metric_col],
                       marker='o', linewidth=2, label=method, 
                       color=color_map[method], markersize=6, alpha=0.8)
        
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Malicious User Ratio (%)', fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8, ncol=2)
    
    plt.tight_layout()
    output_path = Path(output_dir) / '1_malicious_ratio_vs_performance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved: %s", output_path)


def plot_attack_method_vs_malicious_benefit(df, output_dir):
    logger.info("\n" + "=" * 60)
    logger.info("Impact of attack methods on user benefits (malicious and normal users)")
    logger.info("=" * 60)
    
    df_malicious = df[df['Malicious Users'] > 0].copy()
    
    df_normal = df[df['Normal Users'] > 0].copy()
    
    if len(df_malicious) == 0 and len(df_normal) == 0:
        logger.warning("Warning: no user data found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Attack Method Impact on User Benefits (Malicious & Normal Users)', 
                 fontsize=14, fontweight='bold')
    
    all_methods = set()
    if len(df_malicious) > 0:
        all_methods.update(df_malicious['Attack Method'].unique())
    if len(df_normal) > 0:
        all_methods.update(df_normal['Attack Method'].unique())
    attack_methods = [m for m in all_methods if pd.notna(m) and m != '']
    attack_methods = sorted(attack_methods)
    
    max_malicious_requests = df_malicious['Malicious User Requests'].max() if len(df_malicious) > 0 and df_malicious['Malicious User Requests'].notna().any() else 1
    max_normal_requests = df_normal['Normal User Requests'].max() if len(df_normal) > 0 and df_normal['Normal User Requests'].notna().any() else 1
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(attack_methods)))
    
    ax1 = axes[0, 0]
    for idx, method in enumerate(attack_methods):
        method_data = df_malicious[df_malicious['Attack Method'] == method].copy()
        if len(method_data) == 0:
            continue
        
        method_data = method_data[method_data['Malicious User Goodput Rate'].notna() & 
                                 method_data['Malicious User Requests'].notna()]
        if len(method_data) == 0:
            continue
        
        bubble_size = (method_data['Malicious User Requests'].values / max_malicious_requests * 500).clip(50, 1000)
        
        x_pos = idx + np.random.normal(0, 0.1, len(method_data))
        ax1.scatter(x_pos, 
                   method_data['Malicious User Goodput Rate'],
                   s=bubble_size, alpha=0.6, label=method, color=colors[idx], 
                   edgecolors='black', linewidth=0.5)
    
    ax1.set_xticks(range(len(attack_methods)))
    ax1.set_xticklabels(attack_methods, rotation=45, ha='right')
    ax1.set_xlabel('Attack Method', fontsize=11)
    ax1.set_ylabel('Malicious User Goodput Rate', fontsize=11)
    ax1.set_title('Malicious User Goodput by Attack Method\n(Bubble size = Malicious Requests)', 
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc='best', fontsize=8, ncol=2)
    
    ax2 = axes[0, 1]
    for idx, method in enumerate(attack_methods):
        method_data = df_malicious[df_malicious['Attack Method'] == method].copy()
        if len(method_data) == 0:
            continue
        
        method_data = method_data[method_data['Malicious User Average Latency (s)'].notna() & 
                                 method_data['Malicious User Requests'].notna()]
        if len(method_data) == 0:
            continue
        
        bubble_size = (method_data['Malicious User Requests'].values / max_malicious_requests * 500).clip(50, 1000)
        
        x_pos = idx + np.random.normal(0, 0.1, len(method_data))
        ax2.scatter(x_pos, 
                   method_data['Malicious User Average Latency (s)'],
                   s=bubble_size, alpha=0.6, label=method, color=colors[idx],
                   edgecolors='black', linewidth=0.5)
    
    ax2.set_xticks(range(len(attack_methods)))
    ax2.set_xticklabels(attack_methods, rotation=45, ha='right')
    ax2.set_xlabel('Attack Method', fontsize=11)
    ax2.set_ylabel('Malicious User Average Latency (s)', fontsize=11)
    ax2.set_title('Malicious User Latency by Attack Method\n(Bubble size = Malicious Requests)', 
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=8, ncol=2)
    
    ax3 = axes[1, 0]
    for idx, method in enumerate(attack_methods):
        method_data = df_normal[df_normal['Attack Method'] == method].copy()
        if len(method_data) == 0:
            continue
        
        method_data = method_data[method_data['Normal User Goodput Rate'].notna() & 
                                 method_data['Normal User Requests'].notna()]
        if len(method_data) == 0:
            continue
        
        bubble_size = (method_data['Normal User Requests'].values / max_normal_requests * 500).clip(50, 1000)
        
        x_pos = idx + np.random.normal(0, 0.1, len(method_data))
        ax3.scatter(x_pos, 
                   method_data['Normal User Goodput Rate'],
                   s=bubble_size, alpha=0.6, label=method, color=colors[idx], 
                   edgecolors='black', linewidth=0.5)
    
    ax3.set_xticks(range(len(attack_methods)))
    ax3.set_xticklabels(attack_methods, rotation=45, ha='right')
    ax3.set_xlabel('Attack Method', fontsize=11)
    ax3.set_ylabel('Normal User Goodput Rate', fontsize=11)
    ax3.set_title('Normal User Goodput by Attack Method\n(Bubble size = Normal User Requests)', 
                 fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.1)
    ax3.legend(loc='best', fontsize=8, ncol=2)
    
    ax4 = axes[1, 1]
    for idx, method in enumerate(attack_methods):
        method_data = df_normal[df_normal['Attack Method'] == method].copy()
        if len(method_data) == 0:
            continue
        
        method_data = method_data[method_data['Normal User Average Latency (s)'].notna() & 
                                 method_data['Normal User Requests'].notna()]
        if len(method_data) == 0:
            continue
        
        bubble_size = (method_data['Normal User Requests'].values / max_normal_requests * 500).clip(50, 1000)
        
        x_pos = idx + np.random.normal(0, 0.1, len(method_data))
        ax4.scatter(x_pos, 
                   method_data['Normal User Average Latency (s)'],
                   s=bubble_size, alpha=0.6, label=method, color=colors[idx],
                   edgecolors='black', linewidth=0.5)
    
    ax4.set_xticks(range(len(attack_methods)))
    ax4.set_xticklabels(attack_methods, rotation=45, ha='right')
    ax4.set_xlabel('Attack Method', fontsize=11)
    ax4.set_ylabel('Normal User Average Latency (s)', fontsize=11)
    ax4.set_title('Normal User Latency by Attack Method\n(Bubble size = Normal User Requests)', 
                 fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best', fontsize=8, ncol=2)
    
    plt.tight_layout()
    output_path = Path(output_dir) / '2_attack_method_vs_malicious_benefit.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved: %s", output_path)


def plot_dataset_robustness(df, output_dir):
    logger.info("\n" + "=" * 60)
    logger.info("System robustness across different attack datasets")
    logger.info("=" * 60)
    
    df_dataset = df[df['Attack Dataset'].notna() & (df['Attack Dataset'] != '')].copy()
    
    if len(df_dataset) == 0:
        logger.warning("Warning: no attack dataset data found")
        return
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    dataset_summary = df_dataset.groupby(['Attack Method', 'Attack Dataset'])['Overall Goodput Rate'].mean().unstack(fill_value=0)
    dataset_summary.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Overall Goodput Rate by Attack Dataset', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Attack Method', fontsize=10)
    ax1.set_ylabel('Overall Goodput Rate', fontsize=10)
    ax1.legend(title='Attack Dataset', fontsize=9)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)
    
    ax2 = fig.add_subplot(gs[0, 1])
    normal_summary = df_dataset.groupby(['Attack Method', 'Attack Dataset'])['Normal User Goodput Rate'].mean().unstack(fill_value=0)
    normal_summary.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_title('Normal User Goodput Rate by Attack Dataset', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Attack Method', fontsize=10)
    ax2.set_ylabel('Normal User Goodput Rate', fontsize=10)
    ax2.legend(title='Attack Dataset', fontsize=9)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3.3 Heatmap - overall Goodput (attack strength × dataset)
    ax3 = fig.add_subplot(gs[1, 0])
    if 'Attack Strength' in df_dataset.columns and df_dataset['Attack Strength'].notna().any():
        heatmap_data = df_dataset.pivot_table(
            values='Overall Goodput Rate',
            index='Attack Dataset',
            columns='Attack Strength',
            aggfunc='mean'
        )
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=0, vmax=1, ax=ax3, cbar_kws={'label': 'Overall Goodput Rate'})
        ax3.set_title('Heatmap: Overall Goodput (Dataset × Attack Strength)', 
                     fontsize=12, fontweight='bold')
        ax3.set_xlabel('Attack Strength', fontsize=10)
        ax3.set_ylabel('Attack Dataset', fontsize=10)
    else:
        heatmap_data = df_dataset.pivot_table(
            values='Overall Goodput Rate',
            index='Attack Dataset',
            columns='Attack Method',
            aggfunc='mean'
        )
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=0, vmax=1, ax=ax3, cbar_kws={'label': 'Overall Goodput Rate'})
        ax3.set_title('Heatmap: Overall Goodput (Dataset × Attack Method)', 
                     fontsize=12, fontweight='bold')
        ax3.set_xlabel('Attack Method', fontsize=10)
        ax3.set_ylabel('Attack Dataset', fontsize=10)
    
    ax4 = fig.add_subplot(gs[1, 1])
    if 'Attack Strength' in df_dataset.columns and df_dataset['Attack Strength'].notna().any():
        heatmap_data = df_dataset.pivot_table(
            values='Normal User Goodput Rate',
            index='Attack Dataset',
            columns='Attack Strength',
            aggfunc='mean'
        )
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=0, vmax=1, ax=ax4, cbar_kws={'label': 'Normal User Goodput Rate'})
        ax4.set_title('Heatmap: Normal User Goodput (Dataset × Attack Strength)', 
                     fontsize=12, fontweight='bold')
        ax4.set_xlabel('Attack Strength', fontsize=10)
        ax4.set_ylabel('Attack Dataset', fontsize=10)
    else:
        heatmap_data = df_dataset.pivot_table(
            values='Normal User Goodput Rate',
            index='Attack Dataset',
            columns='Attack Method',
            aggfunc='mean'
        )
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=0, vmax=1, ax=ax4, cbar_kws={'label': 'Normal User Goodput Rate'})
        ax4.set_title('Heatmap: Normal User Goodput (Dataset × Attack Method)', 
                     fontsize=12, fontweight='bold')
        ax4.set_xlabel('Attack Method', fontsize=10)
        ax4.set_ylabel('Attack Dataset', fontsize=10)
    
    plt.suptitle('System Robustness Across Different Attack Datasets', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_path = Path(output_dir) / '3_dataset_robustness.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved: %s", output_path)


def plot_total_requests_vs_performance(df, output_dir):
    logger.info("\n" + "=" * 60)
    logger.info("Total requests vs system performance")
    logger.info("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Total Requests vs System Performance', 
                 fontsize=14, fontweight='bold')
    
    ax1 = axes[0]
    x = df['Total Requests'].values
    y = df['Average Latency (s)'].values
    
    scatter = ax1.scatter(x, y, alpha=0.6, s=50, c=df['Malicious Users'].values, 
                         cmap='Reds', edgecolors='black', linewidth=0.5)
    
    if len(x) > 1:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax1.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Fit: y={z[0]:.4f}x+{z[1]:.2f}')
        
        r, p_value = stats.pearsonr(x, y)
        ax1.text(0.05, 0.95, f'r={r:.3f}\np={p_value:.3f}', 
                transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax1.set_xlabel('Total Requests', fontsize=11)
    ax1.set_ylabel('Average Latency (s)', fontsize=11)
    ax1.set_title('Total Requests vs Average Latency', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, label='Malicious Users')
    
    ax2 = axes[1]
    x = df['Total Requests'].values
    y = df['Overall Goodput Rate'].values
    
    scatter = ax2.scatter(x, y, alpha=0.6, s=50, c=df['Malicious Users'].values, 
                         cmap='Blues', edgecolors='black', linewidth=0.5)
    
    if len(x) > 1:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax2.plot(x_line, p(x_line), "b--", alpha=0.8, linewidth=2, label=f'Fit: y={z[0]:.6f}x+{z[1]:.3f}')
        
        r, p_value = stats.pearsonr(x, y)
        ax2.text(0.05, 0.95, f'r={r:.3f}\np={p_value:.3f}', 
                transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2.set_xlabel('Total Requests', fontsize=11)
    ax2.set_ylabel('Overall Goodput Rate', fontsize=11)
    ax2.set_title('Total Requests vs Overall Goodput Rate', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.colorbar(scatter, ax=ax2, label='Malicious Users')
    
    plt.tight_layout()
    output_path = Path(output_dir) / '4_total_requests_vs_performance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved: %s", output_path)


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description='Analyze system performance data')
    parser.add_argument('input_path', type=str, help='Input CSV file or sweep directory')
    parser.add_argument('--sweep-ts', type=str, default=None,
                       help='Filter by sweep timestamp (YYYYMMDD_HHMMSS)')
    parser.add_argument('-o', '--output', type=str, default='analysis_output',
                       help='Output directory (default: analysis_output)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    if not input_path.exists():
        logger.error("Error: path does not exist: %s", input_path)
        return

    if input_path.is_dir():
        run_metas = collect_run_dirs(input_path)
        sweep_ts = args.sweep_ts or pick_latest_sweep_timestamp(run_metas)
        run_metas = filter_by_sweep_timestamp(run_metas, sweep_ts)
        run_dirs = [meta.run_dir for meta in run_metas]
        df = summarize_runs(run_dirs)
        if df.empty:
            logger.warning("No usable experiment results found: %s", input_path)
            return
        if sweep_ts:
            logger.info("Using sweep timestamp: %s", sweep_ts)
    else:
        logger.info("Reading data file: %s", input_path)
        df = pd.read_csv(input_path)

    logger.info("Number of data rows: %d", len(df))
    logger.info("Data columns: %s", list(df.columns))
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)
    
    # Ensure numeric columns have numeric types
    numeric_columns = ['Normal Users', 'Malicious Users', 'Total Requests', 'Average Latency (s)', 
                      'Overall Goodput Rate', 'Normal User Requests', 'Normal User Average Latency (s)',
                      'Normal User Average TTFT (s)', 'Normal User Average TPOT (s)', 'Normal User Average Complete Tokens',
                      'Normal User Goodput Rate', 'Malicious User Requests', 'Malicious User Average Latency (s)', 
                      'Malicious User Average TTFT (s)', 'Malicious User Average TPOT (s)', 'Malicious User Average Complete Tokens',
                      'Malicious User Goodput Rate']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    plot_malicious_ratio_vs_performance(df, output_dir)
    plot_attack_method_vs_malicious_benefit(df, output_dir)
    plot_dataset_robustness(df, output_dir)
    plot_total_requests_vs_performance(df, output_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("Analysis complete! All figures saved to: %s", output_dir)
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
