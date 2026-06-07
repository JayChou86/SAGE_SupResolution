#!/usr/bin/env python
"""娑堣瀺瀹為獙姝ｅ紡鍥?鈥?TWC 鍑虹増璐ㄩ噺 (v2: 鏂囧瓧浣嶇疆淇)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

configs = [
    ("Integer SAGE\n(baseline)",          3.154),
    ("+ Parabolic\ninterpolation",        0.344),
    ("+ 15 iterations\n(from 10)",        0.299),
    ("+ FFT correlation\n(speedup)",      0.338),
    ("+ Local grid\nrefinement",          0.024),
    ("All combined\n(default)",           0.022),
]
labels = [c[0] for c in configs]
values = np.array([c[1] for c in configs])
colors = ['#777777', '#4472C4', '#5B9BD5', '#70AD47', '#ED7D31', '#C00000']

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(range(len(values)), values, color=colors, edgecolor='white',
              linewidth=0.8, width=0.6)

# 鈹€鈹€ Value labels ABOVE bars (not inside) 鈹€鈹€
for i, (bar, val) in enumerate(zip(bars, values)):
    # Place label above the bar
    offset = val * 0.25  # 25% above bar top
    ax.text(i, val + offset, f'{val:.3f}', ha='center', va='bottom',
            fontsize=9, color='#333333')

# 鈹€鈹€ Improvement annotations BELOW each bar (except baseline) 鈹€鈹€
for i in range(1, len(values)):
    impr = values[0] / values[i]
    # Place improvement text centered in the bar 鈥?use light color on dark bars
    bar_color = colors[i]
    text_y = values[i] * 0.5  # middle of bar
    ax.text(i, text_y, f'{impr:.1f}$\\times$', ha='center', va='center',
            fontsize=8, color='white')

# 鈹€鈹€ CRB line 鈹€鈹€
crb_val = 0.023
ax.axhline(crb_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
# Place CRB label at far right, outside bars
ax.text(len(values) - 0.5, crb_val * 1.4, f'CRB = {crb_val:.3f}',
        fontsize=9, color='red', fontstyle='italic', ha='right')

# 鈹€鈹€ Annotation arrows 鈹€鈹€
# Stage 1 annotation: arrow from bar 1 to the side
ax.annotate('Stage 1:\nparabolic interp.\n(9.2$\\times$)',
            xy=(1, values[1]), xytext=(2.6, values[0] * 0.35),
            arrowprops=dict(arrowstyle='->', color='#4472C4', lw=1.5),
            fontsize=8.5, color='#4472C4', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#4472C4', alpha=0.8))

# Stage 2 annotation: arrow from bar 4 to the side
ax.annotate('Stage 2:\nlocal grid refine\n(14.3$\\times$ over Stage 1)',
            xy=(4, values[4]), xytext=(2.6, values[4] * 5),
            arrowprops=dict(arrowstyle='->', color='#ED7D31', lw=1.5),
            fontsize=8.5, color='#ED7D31', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#ED7D31', alpha=0.8))

ax.set_xticks(range(len(values)))
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel('Delay RMSE (samples)', fontsize=12)
ax.set_title('Ablation Study: Contribution of Each Component (0 dB SNR, 100 MC)',
             fontsize=13)
ax.set_yscale('log')
ax.set_ylim([0.005, 12])
ax.grid(axis='y', alpha=0.3)

fig.tight_layout()
results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(results_dir, exist_ok=True)
fig.savefig(os.path.join(results_dir, 'ablation_study.png'), dpi=1200, bbox_inches='tight')
print(f"Saved to {os.path.join(results_dir, 'ablation_study.png')}")
