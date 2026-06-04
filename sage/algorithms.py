"""
核心算法模块
============
实现六种多径时延估计算法:
1. algo_SAGE  — SAGE (整数模式 + 亚采样分数模式)
2. algo_MF    — 匹配滤波器
3. algo_OMP   — 正交匹配追踪
4. algo_NOMP  — Newtonized OMP (连续参数精化)
5. algo_SBL   — 稀疏贝叶斯学习 (含 Woodbury 加速)

所有算法返回 (est_delays, est_amps) 元组。
"""

import numpy as np
from numpy.fft import fft, ifft
from typing import Tuple

from sage.signal_utils import fractional_delay_shift
from sage.interpolation import parabolic_delta


# =========================================================================
# 1. SAGE 算法
# =========================================================================

def algo_SAGE(
    y_rx: np.ndarray, s_base: np.ndarray, N_rx: int, L_paths: int,
    max_iter: int = 15, do_frac: bool = True, use_power: bool = True,
    use_fft_corr: bool = True, local_refine: bool = True,
    refine_step: float = 0.005,
) -> Tuple[np.ndarray, np.ndarray]:
    N_seq = len(s_base)
    Es = np.vdot(s_base, s_base).real
    search_len = N_rx - N_seq
    est_delays = np.zeros(L_paths)
    est_amps = np.zeros(L_paths, dtype=complex)
    residual = y_rx.copy()

    for it in range(max_iter):
        for l in range(L_paths):
            if it > 0:
                s_old = est_amps[l] * fractional_delay_shift(s_base, est_delays[l], N_rx)
                x_l = residual + s_old
            else:
                x_l = residual.copy()

            if use_fft_corr:
                X_f = fft(x_l)
                S_f_conj = np.conj(fft(s_base, n=N_rx))
                corr_abs = np.abs(ifft(X_f * S_f_conj) / Es)[:search_len]
            else:
                corr_abs = np.zeros(search_len)
                for tau in range(search_len):
                    corr_abs[tau] = np.abs(np.vdot(s_base, x_l[tau:tau+N_seq]))

            if it == 0:
                k_opt = int(np.argmax(corr_abs))
            else:
                search_radius = 10
                old_k = int(np.floor(est_delays[l] + 0.5))
                lo = max(0, old_k - search_radius)
                hi = min(search_len - 1, old_k + search_radius)
                if lo < hi:
                    k_opt = lo + int(np.argmax(corr_abs[lo:hi+1]))
                else:
                    k_opt = int(np.argmax(corr_abs))

            if do_frac and 0 < k_opt < search_len - 1:
                if use_power:
                    y_m1 = corr_abs[k_opt-1]**2
                    y_0 = corr_abs[k_opt]**2
                    y_p1 = corr_abs[k_opt+1]**2
                else:
                    y_m1, y_0, y_p1 = corr_abs[k_opt-1], corr_abs[k_opt], corr_abs[k_opt+1]
                delta = parabolic_delta(y_m1, y_0, y_p1)
                tau_est = float(k_opt) + delta
            else:
                tau_est = float(k_opt)

            if local_refine and do_frac:
                tau_est, amp_est = _local_refine(x_l, s_base, tau_est, N_rx, refine_step)
            else:
                s_temp = fractional_delay_shift(s_base, tau_est, N_rx)
                amp_est = np.vdot(s_temp, x_l) / Es

            est_delays[l] = tau_est
            est_amps[l] = amp_est
            residual = x_l - amp_est * fractional_delay_shift(s_base, tau_est, N_rx)

        if it == 0:
            sort_idx = np.argsort(-np.abs(est_amps))
            est_delays = est_delays[sort_idx]
            est_amps = est_amps[sort_idx]
            for i in range(L_paths):
                for j in range(i+1, L_paths):
                    if abs(est_delays[i] - est_delays[j]) < 2.0:
                        if np.abs(est_amps[i]) >= np.abs(est_amps[j]):
                            est_delays[j] += 3.0
                        else:
                            est_delays[i] += 3.0

    return est_delays, est_amps


def _local_refine(x_l, s_base, tau_coarse, N_rx, step=0.005):
    Es = np.vdot(s_base, s_base).real
    best_obj, tau_best, amp_best = -np.inf, tau_coarse, 0j
    for tau_test in np.linspace(max(0, tau_coarse-0.5), tau_coarse+0.5, max(3, int(1.0/step))):
        s_test = fractional_delay_shift(s_base, tau_test, N_rx)
        amp_test = np.vdot(s_test, x_l) / Es
        obj = -np.sum(np.abs(x_l - amp_test*s_test)**2)
        if obj > best_obj:
            best_obj, tau_best, amp_best = obj, tau_test, amp_test
    return tau_best, amp_best


# =========================================================================
# 2. 匹配滤波器
# =========================================================================

def algo_MF(y_rx, A_dict, tau_grid, L_paths):
    corr_mf = np.abs(A_dict.conj().T @ y_rx)
    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(corr_mf, distance=3)
        if len(peaks) >= L_paths:
            idx_mf = peaks[np.argsort(corr_mf[peaks])[::-1]][:L_paths]
        else:
            idx_mf = np.argsort(corr_mf)[::-1][:L_paths]
    except ImportError:
        idx_mf = np.argsort(corr_mf)[::-1][:L_paths]
    est_delays = tau_grid[idx_mf]
    est_amps, _, _, _ = np.linalg.lstsq(A_dict[:, idx_mf], y_rx, rcond=None)
    return est_delays, est_amps


# =========================================================================
# 3. 正交匹配追踪
# =========================================================================

def algo_OMP(y_rx, A_dict, tau_grid, L_paths):
    residual = y_rx.copy()
    idx_omp = []
    for _ in range(L_paths):
        corr_omp = np.abs(A_dict.conj().T @ residual)
        best_idx = int(np.argmax(corr_omp))
        idx_omp.append(best_idx)
        A_sub = A_dict[:, idx_omp]
        amps_omp, _, _, _ = np.linalg.lstsq(A_sub, y_rx, rcond=None)
        residual = y_rx - A_sub @ amps_omp
    idx_omp = np.array(idx_omp)
    est_delays = tau_grid[idx_omp]
    est_amps, _, _, _ = np.linalg.lstsq(A_dict[:, idx_omp], y_rx, rcond=None)
    return est_delays, est_amps


# =========================================================================
# 4. Newtonized OMP (NOMP)
# =========================================================================

def algo_NOMP(
    y_rx, s_base, N_rx, L_paths, tau_grid,
    newton_iters=3, cyclic_iters=1,
):
    N_seq = len(s_base)
    search_len = N_rx - N_seq

    # 频域预计算
    S_f = fft(s_base)

    residual = y_rx.copy()
    active_delays = []

    for l in range(L_paths):
        # 1. 时域粗搜索
        corr_abs = np.zeros(search_len)
        for tau in range(search_len):
            corr_abs[tau] = np.abs(np.vdot(s_base, residual[tau:tau+N_seq]))
        k_opt = int(np.argmax(corr_abs))
        tau_current = float(k_opt)

        # 2. Newton 精化 (窗内)
        win_start = max(0, k_opt - 1)
        win_end = min(N_rx, k_opt + N_seq + 1)
        r_win = residual[win_start:win_end]
        tau_local = tau_current - win_start

        tau_local = _newton_refine(r_win, s_base, tau_local, newton_iters)
        tau_current = win_start + tau_local

        # 3. 冲突检测
        for _retry in range(3):
            too_close = any(abs(tau_current - t) < 3.0 for t in active_delays)
            if not too_close:
                break
            lo, hi = max(0, k_opt-5), min(search_len-1, k_opt+5)
            corr_abs[lo:hi+1] = 0
            if np.max(corr_abs) <= 0:
                tau_current = float(k_opt) + 10.0
                break
            k_opt = int(np.argmax(corr_abs))
            tau_current = float(k_opt)
            w0 = max(0, k_opt-1)
            w1 = min(N_rx, k_opt+N_seq+1)
            r_win = residual[w0:w1]
            tau_current = w0 + _newton_refine(r_win, s_base, tau_current-w0, newton_iters)

        active_delays.append(tau_current)

        # 4. LS + cyclic refinement
        for cyc in range(cyclic_iters + 1):
            A_sub = np.zeros((N_rx, len(active_delays)), dtype=complex)
            for a, t in enumerate(active_delays):
                A_sub[:, a] = fractional_delay_shift(s_base, t, N_rx)
            amps_all, _, _, _ = np.linalg.lstsq(A_sub, y_rx, rcond=None)

            if cyc == cyclic_iters:
                break

            for a in range(len(active_delays)):
                r_other = y_rx.copy()
                for b in range(len(active_delays)):
                    if b != a:
                        r_other -= amps_all[b] * fractional_delay_shift(s_base, active_delays[b], N_rx)
                ka = int(np.floor(active_delays[a]+0.5))
                w0, w1 = max(0, ka-1), min(N_rx, ka+N_seq+1)
                active_delays[a] = w0 + _newton_refine(r_other[w0:w1], s_base, active_delays[a]-w0, newton_iters)

        # 5. 残差更新
        A_sub = np.zeros((N_rx, len(active_delays)), dtype=complex)
        for a, t in enumerate(active_delays):
            A_sub[:, a] = fractional_delay_shift(s_base, t, N_rx)
        amps_all, _, _, _ = np.linalg.lstsq(A_sub, y_rx, rcond=None)
        residual = y_rx - A_sub @ amps_all

    return np.array(active_delays), amps_all


def _newton_refine(r_win, s_base, tau, max_iters=3):
    """频域 Newton 精化: 最大化 |⟨s(τ), r_win⟩|²"""
    N_win = len(r_win)
    R_f = fft(r_win)
    S_w = fft(s_base, n=N_win)
    k_w = np.concatenate([
        np.arange(0, int(np.ceil(N_win/2))),
        np.arange(-int(np.floor(N_win/2)), 0),
    ])
    w = 2*np.pi*k_w/N_win
    Sp = (1j*w)*S_w
    Sd = -(w**2)*S_w

    for _ in range(max_iters):
        ph = np.exp(1j*w*tau)
        c = np.sum(np.conj(S_w)*R_f*ph)/N_win
        cp = np.sum(np.conj(Sp)*R_f*ph)/N_win
        cd = np.sum(np.conj(Sd)*R_f*ph)/N_win
        fp = 2*np.real(np.conj(c)*cp)
        fd = 2*np.real(np.conj(c)*cd + np.abs(cp)**2)
        if abs(fd) < 1e-12:
            break
        delta = np.clip(fp/fd, -0.5, 0.5)
        tau = max(0.0, min(float(N_win), tau-delta))
        if abs(delta) < 1e-6:
            break
    return tau


# =========================================================================
# 5. 稀疏贝叶斯学习 (SBL)
# =========================================================================

def algo_SBL(y_rx, A_dict, tau_grid, L_paths, N0, max_iter=20, use_woodbury=True):
    N_rx, N_grid = A_dict.shape
    gamma_tol = 1e-6
    eps = 1e-6
    gamma = np.abs(A_dict.conj().T @ y_rx)**2 / (N_rx**2)

    for _ in range(max_iter):
        if use_woodbury:
            inv_gamma = np.where(gamma > gamma_tol, 1.0/gamma, 0.0)
            M = N0*np.diag(inv_gamma) + A_dict.conj().T @ A_dict + eps*np.eye(N_grid)
            try:
                inv_M = np.linalg.solve(M, np.eye(N_grid))
            except np.linalg.LinAlgError:
                inv_M = np.linalg.pinv(M)
            Ahy = A_dict.conj().T @ y_rx
            B = A_dict.conj().T @ A_dict
            mu = (gamma/N0)*(Ahy - B @ (inv_M @ Ahy))
            diag_term = np.zeros(N_grid)
            for k in range(N_grid):
                col_k = B[:, k]
                diag_term[k] = (B[k,k].real - np.real(col_k.conj().T @ (inv_M @ col_k)))/N0
        else:
            Gamma = np.diag(gamma)
            Sigma_y = A_dict @ Gamma @ A_dict.conj().T + N0*np.eye(N_rx)
            try:
                inv_Sigma_y = np.linalg.solve(Sigma_y+eps*np.eye(N_rx), np.eye(N_rx))
            except np.linalg.LinAlgError:
                inv_Sigma_y = np.linalg.pinv(Sigma_y+eps*np.eye(N_rx))
            mu = Gamma @ A_dict.conj().T @ inv_Sigma_y @ y_rx
            diag_term = np.sum(A_dict.conj().T*(A_dict.conj().T @ inv_Sigma_y).T, axis=1).real

        gamma_new = np.abs(mu)**2 + gamma - gamma**2*diag_term
        gamma_new = np.maximum(gamma_new, gamma_tol)
        if np.linalg.norm(gamma_new-gamma)/np.linalg.norm(gamma) < 1e-3:
            gamma = gamma_new; break
        gamma = gamma_new

    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(gamma, distance=5)
        if len(peaks) >= L_paths:
            idx_sbl = peaks[np.argsort(gamma[peaks])[::-1]][:L_paths]
        else:
            idx_sbl = np.argsort(gamma)[::-1][:L_paths]
    except ImportError:
        idx_sbl = np.argsort(gamma)[::-1][:L_paths]

    return tau_grid[idx_sbl], mu[idx_sbl]


# =========================================================================
# 6. Root-MUSIC — 子空间方法 / 连续时延估计
# =========================================================================

def algo_ROOTMUSIC(
    y_rx: np.ndarray,
    s_base: np.ndarray,
    L_paths: int,
    N_rx: int = None,
    M_smooth: int = None,
    signal_thresh: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Root-MUSIC 时延估计 (Frequency-Domain)。

    原理:
      频域信道比 H(f) = Y(f)/S(f) ≈ Σ a_k·e^{-j2πfτ_k}
      → 构建平滑 Hankel 矩阵 → SVD → 噪声子空间 → 求根多项式
      → K 个离单位圆最近的根 → 连续时延

    Parameters
    ----------
    y_rx : 1D ndarray
        接收信号 (实或复)。
    s_base : 1D ndarray
        发射信号。
    L_paths : int
        估计的多径数。
    N_rx : int, optional
        接收信号长度 (默认 = len(y_rx))。
    M_smooth : int, optional
        Hankel 矩阵的列数 (默认 = N_rx//3)。
    signal_thresh : float
        频域比值 H(f) 仅在 |S(f)| > thresh·max(|S|) 的频率点计算。

    Returns
    -------
    est_delays : 1D ndarray
        连续时延估计 (采样点)。
    est_amps : 1D ndarray
        复幅度 (最小二乘重估)。
    """
    if N_rx is None:
        N_rx = len(y_rx)

    N_seq = len(s_base)

    # 1. 频域比值 H(f) = Y(f) / S(f)
    # 零填充到 N_rx 以匹配接收信号长度
    S_f_full = np.fft.fft(s_base, n=N_rx)
    Y_f = np.fft.fft(y_rx, n=N_rx)

    # 只在高 SNR 频点使用
    mask = np.abs(S_f_full) > signal_thresh * np.max(np.abs(S_f_full))

    # 对于实信号，只用正频率部分 (避免共轭对称导致 rank 翻倍)
    if np.allclose(y_rx.imag, 0) and np.allclose(s_base.imag, 0):
        # 正频率, 跳过 DC
        freq_start, freq_end = 1, N_rx // 2
        mask = mask[freq_start:freq_end]
        H_raw = Y_f[freq_start:freq_end] / (S_f_full[freq_start:freq_end] + 1e-12)
        delta_f = 1.0 / N_rx
    else:
        # 复信号: 用正半频带 f ∈ [0, Δf, ..., (N_rx/2-1)Δf]
        # 全频带 FFT 的负频率部分引入非单调相位 → 破坏 Vandermonde 结构
        half = N_rx // 2
        H_raw = Y_f[:half] / (S_f_full[:half] + 1e-12)
        delta_f = 1.0 / N_rx

    # 实信号: 应用 mask (带限); 复信号: H_raw 无需 mask
    if np.allclose(y_rx.imag, 0) and np.allclose(s_base.imag, 0):
        H_raw = H_raw[mask]
    M = len(H_raw)
    if M < 3 * L_paths:
        raise ValueError(f"Not enough frequency bins ({M}) for {L_paths} paths")

    # 2. 构建 Hankel 平滑矩阵 (前向-后向平均)
    if M_smooth is None:
        M_smooth = max(L_paths * 4, M // 3)
    L = M - M_smooth + 1  # Hankel 矩阵行数
    if L <= L_paths:
        L = L_paths + 1
        M_smooth = M - L + 1

    H_hankel = np.zeros((L, M_smooth), dtype=complex)
    for i in range(L):
        H_hankel[i, :] = H_raw[i:i + M_smooth]

    # 前向-后向平均 (增强对相干源的鲁棒性)
    H_fb = np.hstack([H_hankel, np.fliplr(np.conj(H_hankel))])

    # 3. SVD → 噪声子空间
    U, s, Vh = np.linalg.svd(H_fb, full_matrices=False)
    # 噪声子空间: 舍弃前 L_paths 个奇异向量
    U_noise = U[:, L_paths:]

    # 4. 求根 MUSIC
    # 多项式系数: p_i = Σ_{k} (U_n U_n^H) 沿对角求和
    # P(z) = a^H(z) U_n U_n^H a(z),  a(z) = [1, z, ..., z^{L-1}]
    # 求根多项式: f(z) = z^{L-1} · a^T(1/z) U_n U_n^H a(z)
    Q = U_noise @ U_noise.conj().T  # L × L

    # 构建求根多项式系数 (degree 2L-2)
    # MUSIC null spectrum: P(z) = a^H(z) Q a(z) = Σ Q_{i,j} z^{j-i}
    # Multiply by z^{L-1}: f(z) = Σ Q_{i,j} z^{L-1+j-i}
    # coeff[k] = Σ_{i,j: L-1+j-i=k} Q_{i,j}
    poly_coeffs = np.zeros(2 * L - 1, dtype=complex)
    for i in range(L):
        for j in range(L):
            k = L - 1 + j - i
            if 0 <= k < 2 * L - 1:
                poly_coeffs[k] += Q[i, j]

    # 求所有根
    roots = np.roots(poly_coeffs)

    # 5. 选 L_paths 个离单位圆最近的互异根 → 转换为时延
    tol = 1e-2
    unit_circle_dist = np.abs(np.abs(roots) - 1.0)
    sorted_idx = np.argsort(unit_circle_dist)

    selected_delays = []
    for idx in sorted_idx:
        z = roots[idx]
        if np.abs(np.abs(z) - 1.0) > 0.05:  # 只取近单位圆根
            continue
        tau = -np.angle(z) / (2 * np.pi * delta_f)
        tau = np.mod(tau, N_rx)
        # 去重: 与前已选时延距离 > 5 采样点
        if all(abs(tau - t) > 5.0 for t in selected_delays):
            selected_delays.append(tau)
        if len(selected_delays) >= L_paths:
            break

    # 不足 K 个时用最接近单位圆的根填充
    while len(selected_delays) < L_paths:
        idx = sorted_idx[len(selected_delays)]
        z = roots[idx]
        tau = np.mod(-np.angle(z) / (2 * np.pi * delta_f), N_rx)
        if all(abs(tau - t) > 1.0 for t in selected_delays):
            selected_delays.append(tau)
        else:
            selected_delays.append(tau + len(selected_delays) * 10.0)

    delays = np.sort(np.array(selected_delays))

    # 6. 最小二乘重估幅度
    # 构建字典: A[:,k] = s_base 延迟 τ_k
    A_dict = np.zeros((N_rx, L_paths), dtype=complex)
    for k in range(L_paths):
        from sage.signal_utils import fractional_delay_shift
        A_dict[:, k] = fractional_delay_shift(s_base, delays[k], N_rx)

    amps, _, _, _ = np.linalg.lstsq(A_dict, y_rx, rcond=None)

    return delays, amps

