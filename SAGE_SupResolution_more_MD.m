% =========================================================================
% 多径时延估计算法 Monte Carlo 性能评估 (RMSE vs SNR)
% 包含：SAGE(整数/亚采样)、MF、OMP、SBL 以及 CRB 理论下界对比
% 带蒙托卡罗仿真
% =========================================================================
clear; clc; close all;

%% 1. 基础参数设置
snr_dB_vec = -15:5:15; % SNR 范围从 -5dB 到 25dB
grid_res = 0.83; % 网格
N_mc = 50; % 蒙特卡洛次数 (如需更平滑的曲线，可设为 200 或 500)
rng(20); % 固定随机种子，保证每次运行结果可复现

s = rng;
current_seed = s.Seed;
fs = 10000; 
N_seq = 511; 
s_tx_base = sign(randn(N_seq, 1)) + 1j * sign(randn(N_seq, 1));
s_tx_base = s_tx_base / norm(s_tx_base);

% 真实信道参数 (避开整数和特定网格点)
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
signal_power = var(y_clean); % 用于计算信噪比

%% 2. 准备细网格字典 (供 MF, OMP, SBL 使用)
% 为了平衡运行时间与网格误差，MC 仿真中网格分辨率设为 0.5 
% (你可以根据算力调整为 0.1，但 MC 会慢几倍)
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
energy_s_prime = norm(s_prime)^2;

%% 4. Monte Carlo 仿真设置

% 初始化 RMSE 记录矩阵 (行: SNR, 列: 多径1,2,3)
rmse_int  = zeros(length(snr_dB_vec), L_paths);
rmse_frac = zeros(length(snr_dB_vec), L_paths);
rmse_mf   = zeros(length(snr_dB_vec), L_paths);
rmse_omp  = zeros(length(snr_dB_vec), L_paths);
rmse_sbl  = zeros(length(snr_dB_vec), L_paths);
crb_bound = zeros(length(snr_dB_vec), L_paths);

fprintf('================================================================\n');
fprintf('开始 Monte Carlo 仿真 (共 %d 个 SNR 点, 每个点 %d 次迭代)...\n', length(snr_dB_vec), N_mc);
fprintf('================================================================\n');

tic; % 计时开始
for snr_idx = 1:length(snr_dB_vec)
    SNR = snr_dB_vec(snr_idx);
    
    % 根据 SNR 计算当前的噪声方差和标准差
    noise_var = signal_power / (10^(SNR/10));
    noise_std = sqrt(noise_var / 2); % 复噪声，实部虚部各分一半方差
    
    % 记录当前 SNR 下的理论 CRB 标准差
    CRB_delays = (noise_std^2 * 2) ./ (2 * abs(true_amps).^2 * energy_s_prime);
    crb_bound(snr_idx, :) = sqrt(CRB_delays); 
    
    % 累加平方误差用的零矩阵
    err_sq_int = zeros(1, L_paths);
    err_sq_frac = zeros(1, L_paths);
    err_sq_mf = zeros(1, L_paths);
    err_sq_omp = zeros(1, L_paths);
    err_sq_sbl = zeros(1, L_paths);
    
    for mc = 1:N_mc
        % 加入当前 SNR 的随机噪声
        noise = noise_std * (randn(N_rx,1) + 1j*randn(N_rx,1));
        y_rx = y_clean + noise; 
        
        %% --- 算法 A: 整数 SAGE ---
        est_delays_int = zeros(L_paths, 1); est_amps_int = zeros(L_paths, 1); 
        residual_int = y_rx;
        for iter = 1:5 % MC中可稍微减少迭代次数提速
            for l = 1:L_paths
                if iter > 1
                    s_old = est_amps_int(l) * fractional_delay_shift(s_tx_base, est_delays_int(l), N_rx);
                    x_l = residual_int + s_old;
                else; x_l = residual_int; end
                
                search_len = N_rx - N_seq;
                corr_vals = zeros(search_len, 1);
                for tau = 1:search_len
                    corr_vals(tau) = abs(s_tx_base' * x_l(tau : tau + N_seq - 1));
                end
                [~, tau_int] = max(corr_vals);
                s_int_temp = fractional_delay_shift(s_tx_base, tau_int, N_rx);
                amp_int = (s_int_temp' * x_l) / (s_tx_base' * s_tx_base);
                est_delays_int(l) = tau_int; est_amps_int(l) = amp_int;
                residual_int = x_l - amp_int * s_int_temp;
            end
        end

        %% --- 算法 B: 小数 SAGE (亚采样) ---
        est_delays_frac = zeros(L_paths, 1); est_amps_frac = zeros(L_paths, 1); 
        residual_frac = y_rx;
        for iter = 1:5
            for l = 1:L_paths
                if iter > 1
                    s_old = est_amps_frac(l) * fractional_delay_shift(s_tx_base, est_delays_frac(l), N_rx);
                    x_l = residual_frac + s_old;
                else; x_l = residual_frac; end
                
                search_len = N_rx - N_seq;
                corr_vals = zeros(search_len, 1);
                for tau = 1:search_len
                    corr_vals(tau) = abs(s_tx_base' * x_l(tau : tau + N_seq - 1));
                end
                [~, k_opt] = max(corr_vals);
                
                if k_opt > 1 && k_opt < search_len
                    y_m1 = corr_vals(k_opt - 1); y_0 = corr_vals(k_opt); y_p1 = corr_vals(k_opt + 1);
                    delta = 0.5 * (y_m1 - y_p1) / (y_m1 - 2*y_0 + y_p1);
                    tau_frac = k_opt + delta;
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

        %% --- 算法 D: OMP ---
        residual_omp_temp = y_rx;
        idx_omp = [];
        for l = 1:L_paths
            corr_omp = abs(A_dict' * residual_omp_temp);
            [~, best_idx] = max(corr_omp);
            idx_omp = [idx_omp, best_idx];
            A_sub = A_dict(:, idx_omp);
            amps_omp = A_sub \ y_rx;
            residual_omp_temp = y_rx - A_sub * amps_omp;
        end
        est_delays_omp = tau_grid(idx_omp)';

        %% --- 算法 E: SBL (稳定版) ---
        gamma = abs(A_dict' * y_rx).^2 / (N_rx^2); 
        gamma_tol = 1e-6;
        sigma2 = noise_var;
        
        for iter_sbl = 1:20 % 控制迭代次数提速
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

        %% --- 路径匹配与误差统计 ---
        for i = 1:L_paths
            [~, id] = min(abs(est_delays_int - true_delays(i))); err_sq_int(i)  = err_sq_int(i) + (est_delays_int(id) - true_delays(i))^2;
            [~, id] = min(abs(est_delays_frac - true_delays(i))); err_sq_frac(i) = err_sq_frac(i) + (est_delays_frac(id) - true_delays(i))^2;
            [~, id] = min(abs(est_delays_mf - true_delays(i)));   err_sq_mf(i)   = err_sq_mf(i) + (est_delays_mf(id) - true_delays(i))^2;
            [~, id] = min(abs(est_delays_omp - true_delays(i)));  err_sq_omp(i)  = err_sq_omp(i) + (est_delays_omp(id) - true_delays(i))^2;
            [~, id] = min(abs(est_delays_sbl - true_delays(i)));  err_sq_sbl(i)  = err_sq_sbl(i) + (est_delays_sbl(id) - true_delays(i))^2;
        end
    end
    
    % 记录计算 RMSE
    rmse_int(snr_idx, :)  = sqrt(err_sq_int / N_mc);
    rmse_frac(snr_idx, :) = sqrt(err_sq_frac / N_mc);
    rmse_mf(snr_idx, :)   = sqrt(err_sq_mf / N_mc);
    rmse_omp(snr_idx, :)  = sqrt(err_sq_omp / N_mc);
    rmse_sbl(snr_idx, :)  = sqrt(err_sq_sbl / N_mc);
    
    fprintf('SNR = %3d dB | MC = %d 次完成 | 小数SAGE RMSE (Path 1) = %.4f | CRB = %.4f\n', ...
        SNR, N_mc, rmse_frac(snr_idx, 1), crb_bound(snr_idx, 1));
end
toc; % 计时结束

%% 5. 绘制 Monte Carlo RMSE 曲线图
% 为了视觉清晰，论文中通常选取【第一条多径 (Path 1)】进行 RMSE 性能展示
target_path = 3;

figure('Color', 'w', 'Position', [150 150 800 600]);
hold on; box on; grid on;

% 绘制各算法 RMSE
lw = 1.8; % 线宽
ms = 8;   % Marker Size
plot(snr_dB_vec, rmse_int(:, target_path),  '-ks', 'LineWidth', lw, 'MarkerSize', ms);
plot(snr_dB_vec, rmse_mf(:, target_path),   '-g*', 'LineWidth', lw, 'MarkerSize', ms);
plot(snr_dB_vec, rmse_omp(:, target_path),  '-md', 'LineWidth', lw, 'MarkerSize', ms);
plot(snr_dB_vec, rmse_sbl(:, target_path),  '-cx', 'LineWidth', lw, 'MarkerSize', ms);
plot(snr_dB_vec, rmse_frac(:, target_path), '-b^', 'LineWidth', 2.5, 'MarkerSize', ms+1); % 突出显示本文算法


% plot(snr_dB_vec, 10*log(rmse_int(:, target_path)),  '-ks', 'LineWidth', lw, 'MarkerSize', ms);
% plot(snr_dB_vec, 10*log(rmse_mf(:, target_path)),   '-g*', 'LineWidth', lw, 'MarkerSize', ms);
% plot(snr_dB_vec, 10*log(rmse_omp(:, target_path)),  '-md', 'LineWidth', lw, 'MarkerSize', ms);
% plot(snr_dB_vec, 10*log(rmse_sbl(:, target_path)),  '-cx', 'LineWidth', lw, 'MarkerSize', ms);
% plot(snr_dB_vec, 10*log(rmse_frac(:, target_path)), '-b^', 'LineWidth', 2.5, 'MarkerSize', ms+1); % 突出显示本文算法
% 绘制 CRB 下界
plot(snr_dB_vec, crb_bound(:, target_path), '--r', 'LineWidth', 2.5);

% 设置 Y 轴为对数坐标系 (RMSE 常用表示法)
set(gca, 'YScale', 'log'); 

% 图形修饰
xlabel('Signal-to-Noise Ratio (SNR) [dB]', 'FontSize', 13, 'FontName', 'Times New Roman');
ylabel('RMSE of Delay Estimation [Samples]', 'FontSize', 13, 'FontName', 'Times New Roman');
title(sprintf('Monte Carlo Simulation (Path %d, %d trials)Seed%d', target_path, N_mc, current_seed), 'FontSize', 14);

% 图例
legend({'Int SAGE (Grid-bound)', 'MF (Grid-bound)', 'OMP (Grid-bound)', 'SBL (Grid-bound)', ...
        'Frac SAGE (Off-grid, Proposed)', 'Cramer-Rao Bound (CRB)'}, ...
        'FontSize', 11, 'Location', 'northeast', 'FontName', 'Times New Roman');

set(gca, 'FontName', 'Times New Roman', 'FontSize', 11);
xlim([min(snr_dB_vec), max(snr_dB_vec)]);

% 如果 y 轴下限过低，可以手动调整以提升美观度
% ylim([min(crb_bound(:, target_path))/2, 10]);



% 添加时间戳确保文件名唯一
timestamp = datestr(now, 'yyyy-mm-dd_HHMMSS');
% 指定文件夹
folder_path = 'D:\A_Matlab_program\A_Matlab_program\SAGE_SupResolution\仿真数据';
if ~exist(folder_path, 'dir')
    mkdir(folder_path);
end
% 保存文件（种子+时间戳）
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