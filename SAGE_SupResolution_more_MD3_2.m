% =========================================================================
% 多径时延估计算法全面性能对比评估框架 (Monte Carlo 仿真版)
% 包含：SAGE(整数/亚采样)、MF、OMP、SBL 以及 CRB 理论下界对比
% 评估指标：时延、幅度、相位的 RMSE 随 SNR 的变化
% 源代码直接修改，优先级高
% =========================================================================
clear; clc; close all;

%% 1. 基础参数设置
SNR_dB_vec = -15:5:15; % 信噪比范围
grid_res = 0.83; % 字典分辨率
num_MC = 30; % Monte Carlo 仿真次数 (建议正式出图设为 200)
% 获取当前随机种子
rng(40); % 固定随机种子，保证每次运行结果可复现
s = rng;
current_seed = s.Seed;


fs = 10000; 
N_seq = 511; 
s_tx_base = sign(randn(N_seq, 1)) + 1j * sign(randn(N_seq, 1));
s_tx_base = s_tx_base / norm(s_tx_base); % 能量归一化

true_delays = [50.43, 60.66, 200.51];
true_amps   = [1.0*exp(1j*0.2), 0.5*exp(-1j*1.5), 0.8*exp(1j*0.8)];
L_paths = length(true_delays);
N_rx = 1000;

% 预生成理想的纯净接收信号
y_clean = zeros(N_rx, 1);
for k = 1:L_paths
    s_frac = fractional_delay_shift(s_tx_base, true_delays(k), N_rx);
    y_clean = y_clean + true_amps(k) * s_frac;
end

%% 2. 准备细网格字典 (供 MF, OMP, SBL 使用，避免在循环内重复生成)
tau_grid = 30:grid_res:220; 
N_grid = length(tau_grid);
A_dict = zeros(N_rx, N_grid);
for g = 1:N_grid
    A_dict(:, g) = fractional_delay_shift(s_tx_base, tau_grid(g), N_rx);
end

%% 3. Monte Carlo 仿真参数设置
num_SNR = length(SNR_dB_vec);
num_MC = 50; % Monte Carlo 仿真次数 (建议正式出图设为 200)

% 算法列表: 1=IntSAGE, 2=FracSAGE, 3=MF, 4=OMP, 5=SBL
num_algos = 5; 
algo_names = {'Integer SAGE', 'Fractional SAGE', 'Matched Filter', 'OMP', 'SBL'};

% 存储 RMSE 结果的矩阵: size = (num_SNR, num_algos)
rmse_delay = zeros(num_SNR, num_algos);
rmse_mag   = zeros(num_SNR, num_algos);
rmse_phase = zeros(num_SNR, num_algos);

% 存储 CRB
crb_delay_all = zeros(num_SNR, 1);
crb_mag_all   = zeros(num_SNR, 1);
crb_phase_all = zeros(num_SNR, 1);

%% 4. CRB 所需的信号导数能量计算
S_freq = fft(s_tx_base);
k_vec = [0:ceil(N_seq/2)-1, -floor(N_seq/2):-1]';
omega = 2 * pi * k_vec / N_seq;
s_prime = ifft((1j * omega) .* S_freq); % 一阶偏导
energy_s_prime = norm(s_prime)^2;
energy_s = norm(s_tx_base)^2; % 已经归一化为1

%% 5. 开始 Monte Carlo 仿真循环
fprintf('开始 Monte Carlo 仿真 (共 %d 个 SNR 点，每个点 %d 次)...\n', num_SNR, num_MC);

for s_idx = 1:num_SNR
    SNR = SNR_dB_vec(s_idx);
    
    % 根据定义计算噪声标准差 (此处 SNR 定义为 10*log10(1/N0) 假设信号均方能量恒定)
    % 复数噪声 w ~ CN(0, N0), 故方差为 N0, 其实部虚部方差各为 N0/2
    N0 = 10^(-SNR/10); 
    noise_std = sqrt(N0 / 2); 
    
    % --- 计算当前 SNR 下的理论 CRB (取多径的平均 CRB 进行作图对比) ---
    CRB_var_delay = N0 ./ (2 * abs(true_amps).^2 * energy_s_prime);
    CRB_var_mag   = N0 ./ (2 * energy_s) * ones(1, L_paths);
    CRB_var_phase = N0 ./ (2 * abs(true_amps).^2 * energy_s);
    
    crb_delay_all(s_idx) = sqrt(mean(CRB_var_delay));
    crb_mag_all(s_idx)   = sqrt(mean(CRB_var_mag));
    crb_phase_all(s_idx) = sqrt(mean(CRB_var_phase));
    
    % --- 初始化误差累加器 ---
    sq_err_delay = zeros(num_algos, 1);
    sq_err_mag   = zeros(num_algos, 1);
    sq_err_phase = zeros(num_algos, 1);
    
    fprintf('  正在处理 SNR = %3d dB... ', SNR);
    
    for mc = 1:num_MC
        % 生成噪声
        noise = noise_std * (randn(N_rx,1) + 1j*randn(N_rx,1));
        y_rx = y_clean + noise;
        
        % 运行各算法
        [d_int, a_int]   = algo_SAGE(y_rx, s_tx_base, N_rx, L_paths, 10, false); % Int SAGE
        [d_frac, a_frac] = algo_SAGE(y_rx, s_tx_base, N_rx, L_paths, 10, true);  % Frac SAGE
        [d_mf, a_mf]     = algo_MF(y_rx, A_dict, tau_grid, L_paths);             % MF
        [d_omp, a_omp]   = algo_OMP(y_rx, A_dict, tau_grid, L_paths);            % OMP
        [d_sbl, a_sbl]   = algo_SBL(y_rx, A_dict, tau_grid, L_paths, N0);        % SBL
        
        % 路径匹配及计算误差
        ests_d = {d_int, d_frac, d_mf, d_omp, d_sbl};
        ests_a = {a_int, a_frac, a_mf, a_omp, a_sbl};
        
        for alg = 1:num_algos
            [m_d, m_a] = match_paths(true_delays, true_amps, ests_d{alg}, ests_a{alg});
            
            % 累加均方误差 (取所有路径的平均)
            sq_err_delay(alg) = sq_err_delay(alg) + mean((m_d - true_delays').^2);
            sq_err_mag(alg)   = sq_err_mag(alg)   + mean((abs(m_a) - abs(true_amps')).^2);
            % 相位误差需考虑卷绕 (wrap to pi)
            phase_err = wrapToPi(angle(m_a) - angle(true_amps'));
            sq_err_phase(alg) = sq_err_phase(alg) + mean(phase_err.^2);
        end
    end
    
    % 计算 RMSE
    rmse_delay(s_idx, :) = sqrt(sq_err_delay / num_MC);
    rmse_mag(s_idx, :)   = sqrt(sq_err_mag / num_MC);
    rmse_phase(s_idx, :) = sqrt(sq_err_phase / num_MC);
    
    fprintf('完成.\n');
end

%% 6. 学术级绘图：RMSE vs SNR
colors = {'k', 'b', 'g', 'm', 'c'};
markers = {'s', '^', '*', 'd', 'x'};
line_styles = {'-', '-', '--', '--', '-'};

figure('Color', 'w', 'Position', [100, 100, 1200, 350]);

% --- 子图1：时延 RMSE ---
subplot(1,3,1);
semilogy(SNR_dB_vec, crb_delay_all, 'r-', 'LineWidth', 2); hold on;
for alg = 1:num_algos
    semilogy(SNR_dB_vec, rmse_delay(:, alg), 'Color', colors{alg}, 'Marker', markers{alg}, ...
        'LineStyle', line_styles{alg}, 'LineWidth', 1.5, 'MarkerSize', 7);
end
grid on; box on;
xlabel('SNR (dB)', 'FontSize', 11); ylabel('Delay RMSE (Samples)', 'FontSize', 11);
title('时延估计 RMSE vs CRB', 'FontSize', 12);
ylim([1e-3, 10]); 
legend(['CRB', algo_names], 'Location', 'southwest', 'FontSize', 8);

% --- 子图2：幅度 RMSE ---
subplot(1,3,2);
semilogy(SNR_dB_vec, crb_mag_all, 'r-', 'LineWidth', 2); hold on;
for alg = 1:num_algos
    semilogy(SNR_dB_vec, rmse_mag(:, alg), 'Color', colors{alg}, 'Marker', markers{alg}, ...
        'LineStyle', line_styles{alg}, 'LineWidth', 1.5, 'MarkerSize', 7);
end
grid on; box on;
xlabel('SNR (dB)', 'FontSize', 11); ylabel('Magnitude RMSE', 'FontSize', 11);
title('幅度估计 RMSE vs CRB', 'FontSize', 12);
legend(['CRB', algo_names], 'Location', 'southwest', 'FontSize', 8);

% --- 子图3：相位 RMSE ---
subplot(1,3,3);
semilogy(SNR_dB_vec, crb_phase_all, 'r-', 'LineWidth', 2); hold on;
for alg = 1:num_algos
    semilogy(SNR_dB_vec, rmse_phase(:, alg), 'Color', colors{alg}, 'Marker', markers{alg}, ...
        'LineStyle', line_styles{alg}, 'LineWidth', 1.5, 'MarkerSize', 7);
end
grid on; box on;
xlabel('SNR (dB)', 'FontSize', 11); ylabel('Phase RMSE (rad)', 'FontSize', 11);
title('相位估计 RMSE vs CRB', 'FontSize', 12);
legend(['CRB', algo_names], 'Location', 'southwest', 'FontSize', 8);

sgtitle('多径参数估计性能全面对比 (Monte Carlo 仿真)', 'FontSize', 14, 'FontWeight', 'bold');





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

%% =========================================================================
% 辅助局部函数区域
% =========================================================================

% 1. SAGE 算法 (包含整数与小数模式)
function [est_delays, est_amps] = algo_SAGE(y_rx, s_base, N_rx, L_paths, max_iter, do_frac)
    N_seq = length(s_base);
    est_delays = zeros(L_paths, 1); est_amps = zeros(L_paths, 1); 
    residual = y_rx;
    search_len = N_rx - N_seq;
    
    for iter = 1:max_iter
        for l = 1:L_paths
            if iter > 1
                s_old = est_amps(l) * fractional_delay_shift(s_base, est_delays(l), N_rx);
                x_l = residual + s_old;
            else
                x_l = residual;
            end
            
            corr_vals = zeros(search_len, 1);
            for tau = 1:search_len
                corr_vals(tau) = abs(s_base' * x_l(tau : tau + N_seq - 1));
            end
            
            [~, k_opt] = max(corr_vals);
            
            if do_frac && k_opt > 1 && k_opt < search_len
                y_m1 = corr_vals(k_opt - 1); y_0 = corr_vals(k_opt); y_p1 = corr_vals(k_opt + 1);
                delta = 0.5 * (y_m1 - y_p1) / (y_m1 - 2*y_0 + y_p1);
                tau_est = k_opt + delta;
            else
                tau_est = k_opt;
            end
            
            s_temp = fractional_delay_shift(s_base, tau_est, N_rx);
            amp_est = (s_temp' * x_l) / (s_base' * s_base);
            
            est_delays(l) = tau_est; est_amps(l) = amp_est;
            residual = x_l - amp_est * s_temp;
        end
    end
end

% 2. 匹配滤波器 (MF)
function [est_delays, est_amps] = algo_MF(y_rx, A_dict, tau_grid, L_paths)
    corr_mf = abs(A_dict' * y_rx);
    [~, locs_mf] = findpeaks(corr_mf, 'SortStr', 'descend');
    if length(locs_mf) < L_paths
        [~, locs_mf] = sort(corr_mf, 'descend');
    end
    idx_mf = locs_mf(1:L_paths);
    est_delays = tau_grid(idx_mf)';
    est_amps = A_dict(:, idx_mf) \ y_rx;
end

% 3. 正交匹配追踪 (OMP)
function [est_delays, est_amps] = algo_OMP(y_rx, A_dict, tau_grid, L_paths)
    residual = y_rx;
    idx_omp = [];
    for l = 1:L_paths
        corr_omp = abs(A_dict' * residual);
        [~, best_idx] = max(corr_omp);
        idx_omp = [idx_omp, best_idx];
        A_sub = A_dict(:, idx_omp);
        amps_omp = A_sub \ y_rx; 
        residual = y_rx - A_sub * amps_omp;
    end
    est_delays = tau_grid(idx_omp)';
    est_amps = amps_omp;
end

% 4. 稀疏贝叶斯学习 (SBL)
function [est_delays, est_amps] = algo_SBL(y_rx, A_dict, tau_grid, L_paths, N0)
    N_rx = length(y_rx);
    gamma = abs(A_dict' * y_rx).^2 / (N_rx^2); 
    gamma_tol = 1e-6;
    
    for iter = 1:20 % MC循环中降低迭代次数以加快速度
        Gamma = diag(gamma);
        Sigma_y = A_dict * Gamma * A_dict' + N0 * eye(N_rx); 
        inv_Sigma_y = inv(Sigma_y + 1e-6 * eye(N_rx));
        mu = Gamma * A_dict' * inv_Sigma_y * y_rx;
        diag_term = sum(A_dict' .* (A_dict' * inv_Sigma_y), 2); 
        gamma_new = abs(mu).^2 + gamma - gamma.^2 .* real(diag_term);
        gamma_new(gamma_new < gamma_tol) = gamma_tol;
        if norm(gamma_new - gamma)/norm(gamma) < 1e-3
            gamma = gamma_new;
            break;
        end
        gamma = gamma_new;
    end
    
    [~, locs_sbl] = findpeaks(gamma, 'SortStr', 'descend', 'MinPeakDistance', 5);
    if length(locs_sbl) < L_paths
        [~, locs_sbl] = sort(gamma, 'descend'); 
    end
    idx_sbl = locs_sbl(1:L_paths);
    est_delays = tau_grid(idx_sbl)';
    est_amps = mu(idx_sbl);
end

% 5. 路径乱序匹配函数
function [matched_delays, matched_amps] = match_paths(t_delays, t_amps, e_delays, e_amps)
    L = length(t_delays);
    matched_delays = zeros(L, 1);
    matched_amps = zeros(L, 1);
    for i = 1:L
        [~, id] = min(abs(e_delays - t_delays(i)));
        matched_delays(i) = e_delays(id);
        matched_amps(i) = e_amps(id);
    end
end

% 6. 分数阶延时平移函数
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