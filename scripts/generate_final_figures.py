#!/usr/bin/env python
"""
生成全部最终论文图片，整合自各脚本。
用法: python scripts/generate_final_figures.py
输出: results/
"""
import sys, os, shutil, time
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 15,
    'mathtext.fontset': 'stix',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
RESULTS = os.path.join(ROOT, 'results')
os.makedirs(RESULTS, exist_ok=True)

def _add_sublabels(axes, ox=-0.08, oy=1.02, fontsize=12):
    for i, ax in enumerate(axes):
        ax.text(ox, oy, '(%s)' % chr(ord('a') + i),
                transform=ax.transAxes, fontsize=fontsize,
                fontweight='bold', ha='left', va='bottom')

t0 = time.time()
print("=" * 60)
print(" Generating final paper figures")
print("=" * 60)

# ═══ 1. fig_01_grid_mismatch ═══
print("[ 1] fig_01_grid_mismatch...")
from generate_all_figures import fig_01; fig_01()

# ═══ 2. flowchart_tikz.pdf (external) ═══
print("[ 2] flowchart_tikz.pdf (copy)...")
src = os.path.join(ROOT, 'SigPro', 'figs', 'flowchart_tikz.pdf')
if os.path.exists(src):
    shutil.copy(src, os.path.join(RESULTS, 'flowchart_tikz.pdf'))

# ═══ 3. fig_03_v12_final ═══
print("[ 3] fig_03_v12_final...")
import gen_fig03_final3

# ═══ 4. fig_04_localrefine ═══
print("[ 4] fig_04_localrefine...")
from generate_paper_figures import generate_localrefine
generate_localrefine()
# rename output
src_lr = os.path.join(RESULTS, 'fig_localrefine.png')
dst_lr = os.path.join(RESULTS, 'fig_04_localrefine.png')
if os.path.exists(src_lr):
    os.replace(src_lr, dst_lr)

# ═══ 5. fig_05_autocorrelation ═══
print("[ 5] fig_05_autocorrelation...")
from generate_all_figures import fig_05; fig_05()

# ═══ 6. fig_06_variance ═══
print("[ 6] fig_06_variance...")
from generate_all_figures import fig_06; fig_06()

# ═══ 7. fig_07_bias_efficiency ═══
print("[ 7] fig_07_bias_efficiency...")
from generate_paper_figures import generate_bias_efficiency
generate_bias_efficiency()

# ═══ 8. fig_09_channel_stem_mat (copy) ═══
print("[ 8] fig_09_channel_stem_mat (copy)...")
for fname in ['fig_09_channel_stem_mat.png', 'fig_09_channel_stem.png']:
    src = os.path.join(ROOT, 'SigPro', 'figs', fname)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(RESULTS, 'fig_09_channel_stem_mat.png'))
        break

# ═══ 9. fig_11_ablation ═══
print("[ 9] fig_11_ablation...")
from generate_all_figures import fig_11; fig_11()

# ═══ 10. fig_12_seatrial_mat (copy) ═══
print("[10] fig_12_seatrial_mat (copy)...")
src_mat = os.path.join(ROOT, 'SigPro', 'figs', 'fig_12_seatrial_mat.png')
if not os.path.exists(src_mat):
    src_mat = os.path.join(ROOT, 'results', 'seatrial_mse.png')
if os.path.exists(src_mat):
    shutil.copy(src_mat, os.path.join(RESULTS, 'fig_12_seatrial_mat.png'))

# ═══ 11-14. RMSE figures from NPZ data ═══
print("[11] fig_rmse_overall + per-path...")
npz_path = os.path.join(ROOT, 'results', '500MC', 'mc_results.npz')
if os.path.exists(npz_path):
    d = np.load(npz_path)
    SNR_dB_vec = d['SNR_dB_vec']; rmse_d = d['rmse_delay']; rmse_m = d['rmse_mag']; rmse_p = d['rmse_phase']
    per_d = d['per_path_delay']; per_m = d['per_path_mag']; per_p = d['per_path_phase']
    crb_d = d['crb_delay']; crb_m = d['crb_mag']; crb_p = d['crb_phase']
    an = list(d['algo_names']); td = d['true_delays']; L = len(td); nA = len(an)
    colors_r = ['#333333','#1f77b4','#2ca02c','#d62728','#ff7f0e','#17becf']
    markers_r = ['s','^','*','d','v','x']; ls_r = ['-','-','--','--','-.','-']
    ylabels = ['Delay RMSE ($T_s$)', 'Magnitude RMSE', 'Phase RMSE (rad)']

    for p in range(L):
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        _add_sublabels(axes)
        for ax_idx, (ax, title, dat, cb) in enumerate(zip(axes,
                ['Delay RMSE', 'Magnitude RMSE', 'Phase RMSE'],
                [per_d[:,:,p], per_m[:,:,p], per_p[:,:,p]],
                [crb_d[:,p], crb_m[:,p], crb_p[:,p]])):
            for a in range(nA):
                ax.semilogy(SNR_dB_vec, dat[:,a], color=colors_r[a], marker=markers_r[a],
                           linestyle=ls_r[a], lw=1.8, ms=7, label=an[a], markerfacecolor='none')
            ax.semilogy(SNR_dB_vec, cb, 'r-', lw=2.5, label='CRB', zorder=10)
            ax.set_xlabel('SNR (dB)'); ax.set_ylabel(ylabels[ax_idx]); ax.set_title(title)
            ax.grid(alpha=0.3, which='both'); ax.set_xlim([-17, 17])
        axes[0].legend(fontsize=11, loc='upper right')  # single legend for figure
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, f'fig_rmse_path{p+1}.png'), dpi=1200, bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    _add_sublabels(axes)
    crb_avg = [np.sqrt(np.mean(crb_d**2, axis=1)), np.sqrt(np.mean(crb_m**2, axis=1)), np.sqrt(np.mean(crb_p**2, axis=1))]
    for ax_idx, (ax, title, dat, cb) in enumerate(zip(axes,
            ['Delay RMSE', 'Magnitude RMSE', 'Phase RMSE'],
            [rmse_d, rmse_m, rmse_p], crb_avg)):
        for a in range(nA):
            ax.semilogy(SNR_dB_vec, dat[:,a], color=colors_r[a], marker=markers_r[a],
                       linestyle=ls_r[a], lw=1.8, ms=7, label=an[a], markerfacecolor='none')
        ax.semilogy(SNR_dB_vec, cb, 'r-', lw=2.5, label='CRB', zorder=10)
        ax.set_xlabel('SNR (dB)'); ax.set_ylabel(ylabels[ax_idx]); ax.set_title(title)
        ax.grid(alpha=0.3, which='both'); ax.set_xlim([-17, 17])
    axes[0].legend(fontsize=11, loc='upper right')  # single legend for figure
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'fig_rmse_overall.png'), dpi=1200, bbox_inches='tight')
    plt.close(fig)
else:
    print("       WARNING: 500MC/mc_results.npz not found")

# ═══ 15. mc_NOF1_paper from NPZ data ═══
print("[15] mc_NOF1_paper...")
wm_dir = os.path.join(ROOT, 'results', 'watermark')
an_wm = ['IntSAGE','CD-SAGE','MF','OMP','NOMP','SBL']
colors_wm = ['#333333','#1f77b4','#2ca02c','#d62728','#ff7f0e','#17becf']
markers_wm = ['s','^','*','d','v','x']; ls_wm = ['-','-','--','--','-.','-']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
_add_sublabels(axes)
for mi, M in enumerate([4, 8]):
    npz_f = os.path.join(wm_dir, f'mc_NOF1_M{M}.npz')
    if os.path.exists(npz_f):
        d = np.load(npz_f)
        ax = axes[mi]
        for a in range(len(an_wm)):
            ax.semilogy(d['SNR'], d['rmse'][:,a], color=colors_wm[a], marker=markers_wm[a],
                       linestyle=ls_wm[a], lw=1.8, ms=7, label=an_wm[a], markerfacecolor='none')
        ax.set_title(f'M={M}', fontsize=12)
        ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Delay RMSE (ms)')
        ax.grid(alpha=0.3, which='both')
    axes[0].legend(fontsize=11, loc='best')  # single legend for figure
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, 'mc_NOF1_paper.png'), dpi=200, bbox_inches='tight')
plt.close(fig)

print(f"\nDone — {time.time()-t0:.0f}s — figures in {RESULTS}/")
