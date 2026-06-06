#!/usr/bin/env python
"""消融实验正式图 — TWC 出版质量"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── 数据 (from STATUS.md Section 2.5, 0 dB SNR) ──
configs = [
    ("Integer SAGE\n(baseline)",          3.154),
    ("+ Parabolic\ninterpolation",        0.344),
    ("+ 15 iterations\n(from 10)",        0.299),
    ("+ FFT correlation\n(speedup)",      0.338),
    ("+ Local grid\nrefinement",          0.024),
    ("All combined\n(default)",           0.022),
]
labels = [c[0] for c in configs]
values = [c[1] for c in configs]
colors = ['#777777', '#4472C4', '#5B9BD5', '#70AD47', '#ED7D31', '#C00000']

fig, ax = plt.subplots(figsize=(8, 4.5))

bars = ax.bar(range(len(values)), values, color=colors, edgecolor='white', linewidth=0.8, width=0.6)

# 数值标注 (对数尺度适配)
for i, (bar, val) in enumerate(zip(bars, values)):
    if val > 1:
        ax.text(i, val * 1.15, f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    else:
        ax.text(i, val * 1.5, f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    # 改善倍数
    if i > 0:
        impr = values[0] / val
        ax.text(i, val * 0.5, f'{impr:.1f}×', ha='center', va='center', fontsize=8, color='white', fontweight='bold')

# CRB 参考线
ax.axhline(0.023, color='red', linestyle='--', linewidth=1.2, alpha=0.7)
ax.text(len(values)-0.6, 0.023*1.3, 'CRB = 0.023', fontsize=9, color='red', fontstyle='italic')

ax.set_xticks(range(len(values)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('Delay RMSE (samples)', fontsize=11)
ax.set_title('Ablation Study: Contribution of Each Component (0 dB SNR)', fontsize=12, fontweight='bold')
ax.set_yscale('log')
ax.set_ylim([0.005, 10])
ax.grid(axis='y', alpha=0.3)

# 注释框
ax.annotate('Stage 1: parabolic\ninterpolation (9×)',
            xy=(1, 0.344), xytext=(2.5, 0.8),
            arrowprops=dict(arrowstyle='->', color='#4472C4'),
            fontsize=8, color='#4472C4')
ax.annotate('Stage 2: local grid\nrefinement (14×)',
            xy=(4, 0.024), xytext=(2.5, 0.06),
            arrowprops=dict(arrowstyle='->', color='#ED7D31'),
            fontsize=8, color='#ED7D31')

fig.tight_layout()
results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(results_dir, exist_ok=True)
fig.savefig(os.path.join(results_dir, 'ablation_study.png'), dpi=200, bbox_inches='tight')
print(f"Saved to {os.path.join(results_dir, 'ablation_study.png')}")
