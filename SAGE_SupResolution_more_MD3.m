% =========================================================================
% 多径估计算法 Monte Carlo 性能评估 (RMSE vs SNR)
% 包含：时延、幅值、相位 全面评估 以及 CRB 理论下界对比
% =========================================================================
clear; clc; close all;

%% 1. 基础参数设置
snr_dB_vec = -15:5:15; % SNR 范围 
grid_res = 1;       % 网格分辨率
N_mc = 200;            % 蒙特卡洛次数 
% rng(1); % 固定随机种子

s = rng;
current_seed = s.Seed;
fs = 10000; 
N_seq = 511; 
s_tx_base = sign(randn(N_seq, 1)) + 1j * sign(randn(N_seq, 1));
s_tx_base = s_tx_base / norm(s_tx_base);

% 真实信道参数 
true_delays = [50.43, 60.66, 200.51];
true_amps   = [1.0*exp(1j*0.2), 0.5*exp(-1j*1.5), 0.8*exp(1j*0.8)];
L_paths = length(true_delays);

N_rx = 1000;

% 生成纯净接收信号 (无噪声)
y_clean = zeros(N_rx, 1);
for k = 1:L_paths
    s_frac = fractional_delay_shift(s_tx_base, true_delays(k), N_rx);
    y_clean = y_clean + true_amps(k) * s_frac;
end
signal_power = var(y_clean); 

%% 2. 准备细网格字典
tau_grid = 30:grid_res:220; 
N_grid = length(tau_grid);
A_dict = zeros(N_rx, N_grid);
for g = 1:N_grid
    A_dict(:, g) = fractional_delay_shift(s_tx_base, tau_grid(g), N_rx);
end

%% 3. 计算理论 CRB 相关的固定项
S_freq = fft(s_tx_base);
k_vec = [0:ceil(N_seq/2)-1, -floor(N_seq/2):-1]';
omega = 2 * pi * k_vec / N_seq;
s_prime = ifft((1j * omega) .* S_freq); 
energy_s_prime = norm(s_prime)^2; % 应该约等于 1，因为 s_tx_base 被归一化了

%% 4. Monte Carlo 仿真设置
% --- 初始化 时延 RMSE ---
rmse_int  = zeros(length(snr_dB_vec), L_paths);
rmse_frac = zeros(length(snr_dB_vec), L_paths);
rmse_mf   = zeros(length(snr_dB_vec), L_paths);
rmse_omp  = zeros(length(snr_dB_vec), L_paths);
rmse_sbl  = zeros(length(snr_dB_vec), L_paths);
crb_bound = zeros(length(snr_dB_vec), L_paths);

% --- 初始化 幅值 RMSE ---
rmse_amp_int  = zeros(length(snr_dB_vec), L_paths);
rmse_amp_frac = zeros(length(snr_dB_vec), L_paths);
rmse_amp_mf   = zeros(length(snr_dB_vec), L_paths);
rmse_amp_omp  = zeros(length(snr_dB_vec), L_paths);
rmse_amp_sbl  = zeros(length(snr_dB_vec), L_paths);
crb_amp_bound = zeros(length(snr_dB_vec), L_paths);

% --- 初始化 相位 RMSE ---
rmse_pha_int  = zeros(length(snr_dB_vec), L_paths);
rmse_pha_frac = zeros(length(snr_dB_vec), L_paths);
rmse_pha_mf   = zeros(length(snr_dB_vec), L_paths);
rmse_pha_omp  = zeros(length(snr_dB_vec), L_paths);
rmse_pha_sbl  = zeros(length(snr_dB_vec), L_paths);
crb_pha_bound = zeros(length(snr_dB_vec), L_paths);

fprintf('================================================================\n');
fprintf('开始 Monte Carlo 仿真 (时延/幅值/相位), 共 %d 个 SNR 点...\n', length(snr_dB_vec));
fprintf('================================================================\n');

tic; 
for snr_idx = 1:length(snr_dB_vec)
    SNR = snr_dB_vec(snr_idx);
    noise_var = signal_power / (10^(SNR/10));
    noise_std = sqrt(noise_var / 2); 
    
    % --- 记录当前 SNR 下的三项理论 CRB 标准差 ---
    % 1. 时延 CRB
    CRB_delays = noise_var ./ (2 * abs(true_amps).^2 * energy_s_prime);
    crb_bound(snr_idx, :) = sqrt(CRB_delays); 
    % 2. 幅值 CRB (对于正交归一化波形，方差与信号幅度无关)
    CRB_amps = noise_var ./ (2 * 1);
    crb_amp_bound(snr_idx, :) = sqrt(CRB_amps) * ones(1, L_paths);
    % 3. 相位 CRB (受真实幅度平方影响)
    CRB_phas = noise_var ./ (2 * abs(true_amps).^2 * 1);
    crb_pha_bound(snr_idx, :) = sqrt(CRB_phas);
    
    % 累加平方误差矩阵 (时延, 幅度, 相位)
    err_sq_int = zeros(1, L_paths); err_sq_amp_int = zeros(1, L_paths); err_sq_pha_int = zeros(1, L_paths);
    err_sq_frac= zeros(1, L_paths); err_sq_amp_frac= zeros(1, L_paths); err_sq_pha_frac= zeros(1, L_paths);
    err_sq_mf  = zeros(1, L_paths); err_sq_amp_mf  = zeros(1, L_paths); err_sq_pha_mf  = zeros(1, L_paths);
    err_sq_omp = zeros(1, L_paths); err_sq_amp_omp = zeros(1, L_paths); err_sq_pha_omp = zeros(1, L_paths);
    err_sq_sbl = zeros(1, L_paths); err_sq_amp_sbl = zeros(1, L_paths); err_sq_pha_sbl = zeros(1, L_paths);
    
    for mc = 1:N_mc
        noise = noise_std * (randn(N_rx,1) + 1j*randn(N_rx,1));
        y_rx = y_clean + noise; 
        
        %% --- 算法 A & B: SAGE (与您的代码一致) ---
        % (省略未改动的算法主体内部代码，详见完整脚本逻辑)
        % --- 算法 A: 整数 SAGE ---
        est_delays_int = zeros(L_paths, 1); est_amps_int = zeros(L_paths, 1); 
        residual_int = y_rx;
        for iter = 1:5
            for l = 1:L_paths
                if iter > 1
                    s_old = est_amps_int(l) * fractional_delay_shift(s_tx_base, est_delays_int(l), N_rx);
                    x_l = residual_int + s_old;
                else; x_l = residual_int; end
                search_len = N_rx - N_seq; corr_vals = zeros(search_len, 1);
                for tau = 1:search_len; corr_vals(tau) = abs(s_tx_base' * x_l(tau : tau + N_seq - 1)); end
                [~, tau_int] = max(corr_vals);
                s_int_temp = fractional_delay_shift(s_tx_base, tau_int, N_rx);
                amp_int = (s_int_temp' * x_l) / (s_tx_base' * s_tx_base);
                est_delays_int(l) = tau_int; est_amps_int(l) = amp_int;
                residual_int = x_l - amp_int * s_int_temp;
            end
        end

        % --- 算法 B: 小数 SAGE ---
        est_delays_frac = zeros(L_paths, 1); est_amps_frac = zeros(L_paths, 1); 
        residual_frac = y_rx;
        for iter = 1:5
            for l = 1:L_paths
                if iter > 1
                    s_old = est_amps_frac(l) * fractional_delay_shift(s_tx_base, est_delays_frac(l), N_rx);
                    x_l = residual_frac + s_old;
                else; x_l = residual_frac; end
                search_len = N_rx - N_seq; corr_vals = zeros(search_len, 1);
                for tau = 1:search_len; corr_vals(tau) = abs(s_tx_base' * x_l(tau : tau + N_seq - 1)); end
                [~, k_opt] = max(corr_vals);
                if k_opt > 1 && k_opt < search_len
                    y_m1 = corr_vals(k_opt - 1); y_0 = corr_vals(k_opt); y_p1 = corr_vals(k_opt + 1);
                    delta = 0.5 * (y_m1 - y_p1) / (y_m1 - 2*y_0 + y_p1); tau_frac = k_opt + delta;
                else; tau_frac = k_opt; end
                s_frac_temp = fractional_delay_shift(s_tx_base, tau_frac, N_rx);
                amp_frac = (s_frac_temp' * x_l) / (s_tx_base' * s_tx_base);
                est_delays_frac(l) = tau_frac; est_amps_frac(l) = amp_frac;
                residual_frac = x_l - amp_frac * s_frac_temp;
            end
        end

        %% --- 算法 C: 匹配滤波器 (MF) ---
        corr_mf = abs(A_dict' * y_rx);
        [~, locs_mf] = findpeaks(corr_mf, 'SortStr', 'descend', 'MinPeakDistance', 3);
        if length(locs_mf) < L_paths; [~, locs_mf] = sort(corr_mf, 'descend'); end
        idx_mf = locs_mf(1:L_paths);
        est_delays_mf = tau_grid(idx_mf)';
        est_amps_mf = A_dict(:, idx_mf) \ y_rx; % 【新增】提取 MF 复幅度

        %% --- 算法 D: OMP ---
        residual_omp_temp = y_rx; idx_omp = [];
        for l = 1:L_paths
            corr_omp = abs(A_dict' * residual_omp_temp);
            [~, best_idx] = max(corr_omp); idx_omp = [idx_omp, best_idx];
            A_sub = A_dict(:, idx_omp); amps_omp = A_sub \ y_rx;
            residual_omp_temp = y_rx - A_sub * amps_omp;
        end
        est_delays_omp = tau_grid(idx_omp)';
        est_amps_omp = amps_omp; % 【新增】提取 OMP 复幅度

        %% --- 算法 E: SBL ---
        gamma = abs(A_dict' * y_rx).^2 / (N_rx^2); gamma_tol = 1e-6; sigma2 = noise_var;
        for iter_sbl = 1:20
            Gamma = diag(gamma);
            Sigma_y = A_dict * Gamma * A_dict' + sigma2 * eye(N_rx); 
            inv_Sigma_y = inv(Sigma_y + 1e-6 * eye(N_rx));
            mu = Gamma * A_dict' * inv_Sigma_y * y_rx;
            diag_term = sum(A_dict' .* (A_dict' * inv_Sigma_y), 2); 
            gamma_new = abs(mu).^2 + gamma - gamma.^2 .* real(diag_term);
            gamma_new(gamma_new < gamma_tol) = gamma_tol;
            if norm(gamma_new - gamma)/norm(gamma) < 1e-3; break; end
            gamma = gamma_new;
        end
        [~, locs_sbl] = findpeaks(gamma, 'SortStr', 'descend', 'MinPeakDistance', 3);
        if length(locs_sbl) < L_paths; [~, locs_sbl] = sort(gamma, 'descend'); end
        idx_sbl = locs_sbl(1:L_paths);
        est_delays_sbl = tau_grid(idx_sbl)';
        est_amps_sbl = mu(idx_sbl); % 【新增】提取 SBL 复幅度

        %% --- 路径匹配与全维度误差统计 ---
        for i = 1:L_paths
            t_tau = true_delays(i); 
            t_mag = abs(true_amps(i)); 
            t_pha = angle(true_amps(i));

            % 1. Int SAGE
            [~, id] = min(abs(est_delays_int - t_tau)); 
            err_sq_int(i)     = err_sq_int(i) + (est_delays_int(id) - t_tau)^2;
            err_sq_amp_int(i) = err_sq_amp_int(i) + (abs(est_amps_int(id)) - t_mag)^2;
            err_sq_pha_int(i) = err_sq_pha_int(i) + (wrapToPi(angle(est_amps_int(id)) - t_pha))^2;
            
            % 2. Frac SAGE
            [~, id] = min(abs(est_delays_frac - t_tau)); 
            err_sq_frac(i)     = err_sq_frac(i) + (est_delays_frac(id) - t_tau)^2;
            err_sq_amp_frac(i) = err_sq_amp_frac(i) + (abs(est_amps_frac(id)) - t_mag)^2;
            err_sq_pha_frac(i) = err_sq_pha_frac(i) + (wrapToPi(angle(est_amps_frac(id)) - t_pha))^2;
            
            % 3. MF
            [~, id] = min(abs(est_delays_mf - t_tau)); 
            err_sq_mf(i)     = err_sq_mf(i) + (est_delays_mf(id) - t_tau)^2;
            err_sq_amp_mf(i) = err_sq_amp_mf(i) + (abs(est_amps_mf(id)) - t_mag)^2;
            err_sq_pha_mf(i) = err_sq_pha_mf(i) + (wrapToPi(angle(est_amps_mf(id)) - t_pha))^2;
            
            % 4. OMP
            [~, id] = min(abs(est_delays_omp - t_tau)); 
            err_sq_omp(i)     = err_sq_omp(i) + (est_delays_omp(id) - t_tau)^2;
            err_sq_amp_omp(i) = err_sq_amp_omp(i) + (abs(est_amps_omp(id)) - t_mag)^2;
            err_sq_pha_omp(i) = err_sq_pha_omp(i) + (wrapToPi(angle(est_amps_omp(id)) - t_pha))^2;
            
            % 5. SBL
            [~, id] = min(abs(est_delays_sbl - t_tau)); 
            err_sq_sbl(i)     = err_sq_sbl(i) + (est_delays_sbl(id) - t_tau)^2;
            err_sq_amp_sbl(i) = err_sq_amp_sbl(i) + (abs(est_amps_sbl(id)) - t_mag)^2;
            err_sq_pha_sbl(i) = err_sq_pha_sbl(i) + (wrapToPi(angle(est_amps_sbl(id)) - t_pha))^2;
        end
    end
    
    % 记录最终的 RMSE
    rmse_int(snr_idx, :)  = sqrt(err_sq_int / N_mc); rmse_amp_int(snr_idx, :)  = sqrt(err_sq_amp_int / N_mc); rmse_pha_int(snr_idx, :)  = sqrt(err_sq_pha_int / N_mc);
    rmse_frac(snr_idx, :) = sqrt(err_sq_frac / N_mc);rmse_amp_frac(snr_idx, :) = sqrt(err_sq_amp_frac / N_mc);rmse_pha_frac(snr_idx, :) = sqrt(err_sq_pha_frac / N_mc);
    rmse_mf(snr_idx, :)   = sqrt(err_sq_mf / N_mc);  rmse_amp_mf(snr_idx, :)   = sqrt(err_sq_amp_mf / N_mc);  rmse_pha_mf(snr_idx, :)   = sqrt(err_sq_pha_mf / N_mc);
    rmse_omp(snr_idx, :)  = sqrt(err_sq_omp / N_mc); rmse_amp_omp(snr_idx, :)  = sqrt(err_sq_amp_omp / N_mc); rmse_pha_omp(snr_idx, :)  = sqrt(err_sq_pha_omp / N_mc);
    rmse_sbl(snr_idx, :)  = sqrt(err_sq_sbl / N_mc); rmse_amp_sbl(snr_idx, :)  = sqrt(err_sq_amp_sbl / N_mc); rmse_pha_sbl(snr_idx, :)  = sqrt(err_sq_pha_sbl / N_mc);
    
    fprintf('SNR = %3d dB | MC = %d | 小数SAGE RMSE -> 时延: %.4f | 幅值: %.4f | 相位: %.4f\n', ...
        SNR, N_mc, rmse_frac(snr_idx, 1), rmse_amp_frac(snr_idx, 1), rmse_pha_frac(snr_idx, 1));
end
toc; 

%% 5. 绘制 Monte Carlo RMSE 曲线图 (三组图：时延、幅值、相位)
target_path = 3;
lw = 1.8; ms = 8;
legend_strs = {'Int SAGE', 'MF', 'OMP', 'SBL', 'Frac SAGE (Proposed)', 'CRB'};

% ==================== 图 1: 时延 RMSE ====================
figure('Color', 'w', 'Position', [100 200 600 500]);
hold on; box on; grid on;
plot(snr_dB_vec, rmse_int(:, target_path),  '-ks', 'LineWidth', lw, 'MarkerSize', ms);
plot(snr_dB_vec, rmse_mf(:, target_path),   '-g*', 'LineWidth', lw, 'MarkerSize', ms);
plot(snr_dB_vec, rmse_omp(:, target_path),  '-md', 'LineWidth', lw, 'MarkerSize', ms);
plot(snr_dB_vec, rmse_sbl(:, target_path),  '-cx', 'LineWidth', lw, 'MarkerSize', ms);
plot(snr_dB_vec, rmse_frac(:, target_path), '-b^', 'LineWidth', 2.5, 'MarkerSize', ms+1); 
plot(snr_dB_vec, crb_bound(:, target_path), '--r', 'LineWidth', 2.5);
set(gca, 'YScale', 'log', 'FontName', 'Times New Roman', 'FontSize', 11); 
xlabel('SNR [dB]'); ylabel('RMSE of Delay [Samples]'); 
title(sprintf('Delay Estimation (Path %d)', target_path));
legend(legend_strs, 'Location', 'best', 'FontName', 'Times New Roman');

% ==================== 图 2: 幅值 RMSE ====================
figure('Color', 'w', 'Position', [750 200 600 500]);
hold on; box on; grid on;
plot(snr_dB_vec, rmse_amp_int(:, target_path),  '-ks', 'LineWidth', lw, 'MarkerSize', ms);
plot(snr_dB_vec, rmse_amp_mf(:, target_path),   '-g*', 'LineWidth', lw, 'MarkerSize', ms);
plot(snr_dB_vec, rmse_amp_omp(:, target_path),  '-md', 'LineWidth', lw, 'MarkerSize', ms);
plot(snr_dB_vec, rmse_amp_sbl(:, target_path),  '-cx', 'LineWidth', lw, 'MarkerSize', ms);
plot(snr_dB_vec, rmse_amp_frac(:, target_path), '-b^', 'LineWidth', 2.5, 'MarkerSize', ms+1); 
plot(snr_dB_vec, crb_amp_bound(:, target_path), '--r', 'LineWidth', 2.5);
set(gca, 'YScale', 'log', 'FontName', 'Times New Roman', 'FontSize', 11); 
xlabel('SNR [dB]'); ylabel('RMSE of Magnitude'); 
title(sprintf('Magnitude Estimation (Path %d)', target_path));
legend(legend_strs, 'Location', 'best', 'FontName', 'Times New Roman');

% ==================== 图 3: 相位 RMSE ====================
figure('Color', 'w', 'Position', [1400 200 600 500]);
hold on; box on; grid on;
plot(snr_dB_vec, rmse_pha_int(:, target_path),  '-ks', 'LineWidth', lw, 'MarkerSize', ms);
plot(snr_dB_vec, rmse_pha_mf(:, target_path),   '-g*', 'LineWidth', lw, 'MarkerSize', ms);
plot(snr_dB_vec, rmse_pha_omp(:, target_path),  '-md', 'LineWidth', lw, 'MarkerSize', ms);
plot(snr_dB_vec, rmse_pha_sbl(:, target_path),  '-cx', 'LineWidth', lw, 'MarkerSize', ms);
plot(snr_dB_vec, rmse_pha_frac(:, target_path), '-b^', 'LineWidth', 2.5, 'MarkerSize', ms+1); 
plot(snr_dB_vec, crb_pha_bound(:, target_path), '--r', 'LineWidth', 2.5);
set(gca, 'YScale', 'log', 'FontName', 'Times New Roman', 'FontSize', 11); 
xlabel('SNR [dB]'); ylabel('RMSE of Phase [rad]'); 
title(sprintf('Phase Estimation (Path %d)', target_path));
legend(legend_strs, 'Location', 'best', 'FontName', 'Times New Roman');


%% 保存数据
timestamp = datestr(now, 'yyyy-mm-dd_HHMMSS');
folder_path = 'D:\A_Matlab_program\A_Matlab_program\SAGE_SupResolution\仿真数据';
if ~exist(folder_path, 'dir'); mkdir(folder_path); end
filename = sprintf('workspace_seed_%d_%s.mat', current_seed, timestamp);
save(fullfile(folder_path, filename));

%% --- 辅助函数：严格基于 FFT 的分数延迟 ---
function s_shifted = fractional_delay_shift(s_base, delay, N_total)
    N_orig = length(s_base);
    S_freq = fft(s_base);
    k = [0:ceil(N_orig/2)-1, -floor(N_orig/2):-1]';
    phase_shift = exp(-1j * 2 * pi * k * (delay - floor(delay)) / N_orig);
    s_temp = ifft(S_freq .* phase_shift);
    s_shifted = zeros(N_total, 1);
    start_idx = floor(delay);
    if start_idx >= 1 && start_idx + N_orig - 1 <= N_total
        s_shifted(start_idx : start_idx + N_orig - 1) = s_temp;
    end
end