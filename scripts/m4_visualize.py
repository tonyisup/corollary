#!/usr/bin/env python3
"""
M4 vs No-M4 Visualization: Generate bar charts comparing frame distributions.
Creates:
1. Text-based bar chart (terminal output)
2. Data file for external plotting (JSON)
3. Simple ASCII chart using matplotlib (if available)
"""
from __future__ import annotations

import json
import os
from collections import Counter


def load_comparison_data(json_path: str) -> dict:
    """Load comparison results from JSON."""
    with open(json_path, "r") as f:
        return json.load(f)


def print_text_chart(data: dict) -> None:
    """Print a text-based bar chart comparing M4 vs No-M4."""
    baseline = data.get("baseline", {})
    m4_on = data.get("m4_on", {})
    
    print("\n" + "="*60)
    print("M4 vs No-M4 FRAME DISTRIBUTION")
    print("="*60 + "\n")
    
    # Aggregate data
    categories = ["SELF", "BOTH", "WORLD"]
    baseline_counts = [baseline.get("total_self", 0), baseline.get("total_both", 0), baseline.get("total_world", 0)]
    m4_counts = [m4_on.get("total_self", 0), m4_on.get("total_both", 0), m4_on.get("total_world", 0)]
    
    max_count = max(max(baseline_counts), max(m4_counts))
    bar_width = 40
    
    print(f"{'Category':<10} {'No-M4':<15} {'M4 ON':<15} {'Improvement':<15}")
    print("-" * 60)
    
    for i, cat in enumerate(categories):
        base = baseline_counts[i]
        m4 = m4_counts[i]
        
        # Bar chart
        base_bar = "#" * int((base / max_count) * bar_width) if max_count > 0 else ""
        m4_bar = "#" * int((m4 / max_count) * bar_width) if max_count > 0 else ""
        
        # Improvement
        if cat == "SELF":
            diff = m4 - base
            pct = (diff / max(base, 1)) * 100
            improvement = f"+{diff} ({pct:+.1f}%)"
        elif cat == "BOTH":
            diff = m4 - base
            pct = (diff / max(base, 1)) * 100
            improvement = f"{diff} ({pct:+.1f}%)"
        else:
            improvement = f"{m4 - base:+d}"
        
        print(f"{cat:<10} {base:>5} {base_bar:<{bar_width}} | {m4:>5} {m4_bar:<{bar_width}} | {improvement:<15}")
    
    print("\n" + "-"*60)
    print(f"M4 Accepted Frames: {m4_on.get('m4_accepted', 0)}")
    print(f"Avg M4 Confidence: {m4_on.get('avg_m4_confidence', 0):.3f}")
    print("="*60)


def save_plot_data(data: dict, output_path: str) -> None:
    """Save structured data for external plotting tools."""
    baseline = data.get("baseline", {})
    m4_on = data.get("m4_on", {})
    
    plot_data = {
        "categories": ["SELF", "BOTH", "WORLD"],
        "no_m4": [baseline.get("total_self", 0), baseline.get("total_both", 0), baseline.get("total_world", 0)],
        "m4_on": [m4_on.get("total_self", 0), m4_on.get("total_both", 0), m4_on.get("total_world", 0)],
        "m4_accepted": m4_on.get("m4_accepted", 0),
        "avg_m4_confidence": m4_on.get("avg_m4_confidence", 0),
    }
    
    with open(output_path, "w") as f:
        json.dump(plot_data, f, indent=2)
    
    print(f"\nSaved plot data to: {output_path}")


def try_matplotlib_chart(data: dict, output_path: str) -> bool:
    """Try to generate a matplotlib chart (if available)."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        baseline = data.get("baseline", {})
        m4_on = data.get("m4_on", {})
        
        categories = ["SELF", "BOTH", "WORLD"]
        no_m4 = [baseline.get("total_self", 0), baseline.get("total_both", 0), baseline.get("total_world", 0)]
        m4_on = [m4_on.get("total_self", 0), m4_on.get("total_both", 0), m4_on.get("total_world", 0)]
        
        x = np.arange(len(categories))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, no_m4, width, label='No M4', color='#ff7f0e', alpha=0.8)
        bars2 = ax.bar(x + width/2, m4_on, width, label='M4 ON', color='#2ca02c', alpha=0.8)
        
        ax.set_ylabel('Frame Count', fontsize=12)
        ax.set_title('M4 vs No-M4 Frame Distribution\n(Corollary Project - M3 Embodied Loop)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"\nSaved matplotlib chart to: {output_path}")
        return True
        
    except ImportError:
        print("\nMatplotlib not available — skipping chart generation.")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize M4 vs No-M4 comparison")
    parser.add_argument("--input", default="data/m4_vs_nom4.json", help="Input comparison JSON")
    parser.add_argument("--output-dir", default="data/visualizations", help="Output directory")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found. Run m4_comparison.py first.")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    data = load_comparison_data(args.input)
    
    # 1. Text-based chart
    print_text_chart(data)
    
    # 2. Save plot data
    plot_data_path = os.path.join(args.output_dir, "m4_comparison_data.json")
    save_plot_data(data, plot_data_path)
    
    # 3. Try matplotlib chart
    chart_path = os.path.join(args.output_dir, "m4_vs_nom4_chart.png")
    try_matplotlib_chart(data, chart_path)
    
    print(f"\nVisualization files saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
