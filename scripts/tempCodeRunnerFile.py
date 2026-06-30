def fig_04():
    tau_true = 50.43
    sigma_c = 0.18
    np.random.seed(42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Coarse
    tau_c = np.arange(49.6, 51.45, 0.05)
    f_c = np.exp(-(tau_c - tau_true)**2/(2*sigma_c**2)) + 0.015*np.random.randn(len(tau_c))
    ax1.plot(tau_c, f_c, 'bo-', ms=7, lw=1.5, label='Coarse grid ($\\Delta=0.05$)')
    ax1.axvline(tau_true, color='green', ls='--', lw=2, label=f'True $\\tau={tau_true:.2f}$')
    idx_c = np.argmax(f_c)
    ax1.axvline(tau_c[idx_c], color='blue', ls=':', lw=2, label=f'$\\tau_c={tau_c[idx_c]:.2f}$')
    ax1.plot(tau_c[idx_c], f_c[idx_c], 'r*', ms=15, zorder=5)
    sh_l, sh_h = tau_c[idx_c]-0.06, tau_c[idx_c]+0.06
    ax1.axvspan(sh_l, sh_h, alpha=0.15, color='orange', label='Fine zone')
    ax1.set_xlabel('Delay (samples)'); ax1.set_ylabel('$|C(\\tau)|^2$')
    ax1.set_title('(a) Pass 1: Coarse Search ($\\Delta=0.05$)')
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # (b) Fine
    tau_f = np.arange(sh_l, sh_h+0.001, 0.005)
    f_f = np.exp(-(tau_f - tau_true)**2/(2*sigma_c**2)) + 0.008*np.random.randn(len(tau_f))
    ax2.plot(tau_f, f_f, 'r-o', ms=5, lw=1.5, label='Fine grid ($\\Delta=0.005$)')
    ax2.axvline(tau_true, color='green', ls='--', lw=2, label=f'True $\\tau={tau_true:.2f}$')
    idx_f = np.argmax(f_f)
    ax2.axvline(tau_f[idx_f], color='darkred', ls=':', lw=2, label=f'$\\tau^*={tau_f[idx_f]:.3f}$')
    ax2.plot(tau_f[idx_f], f_f[idx_f], 'r*', ms=15, zorder=5)
    ax2.set_xlabel('Delay (samples)'); ax2.set_ylabel('$|C(\\tau)|^2$')
    ax2.set_title('(b) Pass 2: Fine Search ($\\Delta=0.005$)')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    #fig.suptitle('Stage 2: Local Grid Refinement', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'fig_04_localrefine.png'), dpi=1200, bbox_inches='tight')
    plt.close(fig)
    print("  鉁?fig_04_localrefine.png")