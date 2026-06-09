% =========================================================================
% 多径时延估计算法全面性能对比评估框架
% 包含：SAGE(整数/亚采样)、MF、OMP、SBL 以及 CRB 理论下界对比
% =========================================================================
clear; clc; close all;

%% 1. 生成带小数时延的仿真信号 (物理真实信道)
fs = 10000; 
N_seq = 511; 
rng(42); % 固定随机种子
grid_res = 1; % 字典分辨率 0.1 采样点

s_tx_base = sign(randn(N_seq, 1)) + 1j * sign(randn(N_seq, 1));
s_tx_base = s_tx_base / norm(s_tx_base);

% 设置带小数点的真实时延，以及复数幅度(包含幅值和相位)
true_delays = [50.4, 65.7, 200.2]; 
true_delays = [50.43, 60.66, 200.51];
true_amps   = [1.0*exp(1j*0.2), 0.5*exp(-1j*1.5), 0.8*exp(1j*0.8)];
L_paths = length(true_delays);

N_rx = 1000;
y_rx = zeros(N_rx, 1);

% 模拟接收信号 (理想的分数延迟)
for k = 1:L_paths
    s_frac = fractional_delay_shift(s_tx_base, true_delays(k), N_rx);
    y_rx = y_rx + true_amps(k) * s_frac;
end
noise_std = 0.05;
noise = noise_std * (randn(N_rx,1) + 1j*randn(N_rx,1));
y_rx = y_rx + noise; 

%% 1.5 计算 Cramer-Rao Bound (CRB) 下界
% 基于离散信号导数能量计算多径时延的理论最优误差标准差
S_freq = fft(s_tx_base);
k_vec = [0:ceil(N_seq/2)-1, -floor(N_seq/2):-1]';
omega = 2 * pi * k_vec / N_seq;
s_prime = ifft((1j * omega) .* S_freq); % 信号关于时延的一阶偏导
energy_s_prime = norm(s_prime)^2;

% 在多径不严重混叠的前提下，各路径时延CRB的独立近似计算
CRB_delays = (noise_std^2) ./ (2 * abs(true_amps).^2 * energy_s_prime);
std_CRB = sqrt(CRB_delays); % 理论误差下界 (标准差)

%% 2. 算法 A: 传统 SAGE 算法 (仅整数搜索 - 无亚采样)
max_iter = 10;
est_delays_int = zeros(L_paths, 1); est_amps_int = zeros(L_paths, 1); 
residual_int = y_rx;
for iter = 1:max_iter
    for l = 1:L_paths
        if iter > 1
            s_old = est_amps_int(l) * fractional_delay_shift(s_tx_base, est_delays_int(l), N_rx);
            x_l = residual_int + s_old;
        else
            x_l = residual_int;
        end
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

%% 3. 算法 B: 超分辨率 SAGE 算法 (带抛物线插值亚采样)
est_delays_frac = zeros(L_paths, 1); est_amps_frac = zeros(L_paths, 1); 
residual_frac = y_rx;
for iter = 1:max_iter
    for l = 1:L_paths
        if iter > 1
            s_old = est_amps_frac(l) * fractional_delay_shift(s_tx_base, est_delays_frac(l), N_rx);
            x_l = residual_frac + s_old;
        else
            x_l = residual_frac;
        end
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
        else
            tau_frac = k_opt;
        end
        s_frac_temp = fractional_delay_shift(s_tx_base, tau_frac, N_rx);
        amp_frac = (s_frac_temp' * x_l) / (s_tx_base' * s_tx_base);
        est_delays_frac(l) = tau_frac; est_amps_frac(l) = amp_frac;
        residual_frac = x_l - amp_frac * s_frac_temp;
    end
end

%% --- 准备细网格字典 (供 MF, OMP, SBL 使用) ---
% grid_res = 0.1; % 字典分辨率 0.1 采样点
tau_grid = 30:grid_res:220; 
N_grid = length(tau_grid);
A_dict = zeros(N_rx, N_grid);
for g = 1:N_grid
    A_dict(:, g) = fractional_delay_shift(s_tx_base, tau_grid(g), N_rx);
end

%% 4. 算法 C: 匹配滤波器 (Matched Filter, MF)
corr_mf = abs(A_dict' * y_rx);
% 寻找局部极值峰值，防止取到同一主瓣上的邻近点
[~, locs_mf] = findpeaks(corr_mf, 'SortStr', 'descend');
idx_mf = locs_mf(1:L_paths);
est_delays_mf = tau_grid(idx_mf)';
est_amps_mf = A_dict(:, idx_mf) \ y_rx; % 最小二乘提幅度
residual_mf = y_rx - A_dict(:, idx_mf) * est_amps_mf;

%% 5. 算法 D: 正交匹配追踪 (OMP)
residual_omp_temp = y_rx;
idx_omp = [];
for l = 1:L_paths
    corr_omp = abs(A_dict' * residual_omp_temp);
    [~, best_idx] = max(corr_omp);
    idx_omp = [idx_omp, best_idx];
    A_sub = A_dict(:, idx_omp);
    amps_omp = A_sub \ y_rx; % 正交投影
    residual_omp_temp = y_rx - A_sub * amps_omp;
end
est_delays_omp = tau_grid(idx_omp)';
est_amps_omp = amps_omp;
residual_omp = residual_omp_temp;


%% 6. 算法 E: 稀疏贝叶斯学习 (SBL - 鲁棒稳定版)
sigma2 = noise_std^2;
% 【修复1】初始化：利用 MF 能量投影作为先验，而不是全1，大幅加快收敛
gamma = abs(A_dict' * y_rx).^2 / (N_rx^2); 
gamma_tol = 1e-6;

for iter = 1:30 % 增加迭代次数保证收敛
    Gamma = diag(gamma);
    % 【修复2】Woodbury加速: 将N_rx×N_rx求逆降为N_grid×N_grid
    % (sigma2*I + A*Gamma*A')^{-1} = I/sigma2 - A*inv(inv(Gamma)+A'*A/sigma2)*A'/sigma2^2
    G_inv = diag(1./(gamma + 1e-12));  % inv(Gamma) 对角矩阵
    V = G_inv + (A_dict' * A_dict) / sigma2;  % N_grid × N_grid (远小于N_rx × N_rx)
    V_reg = V + 1e-6 * eye(N_grid);
    inv_Sigma_y_A = (A_dict - A_dict * (V_reg \ (A_dict' * A_dict)) / sigma2) / sigma2;
    inv_Sigma_y_y = (y_rx - A_dict * (V_reg \ (A_dict' * y_rx)) / sigma2) / sigma2;

    % 均值更新
    mu = Gamma * A_dict' * inv_Sigma_y_y;

    % 协方差更新: diag(A' * inv_Sigma_y * A)
    ATA_s2 = (A_dict' * A_dict) / sigma2;
    diag_term = diag(ATA_s2 - ATA_s2 * (V_reg \ ATA_s2)); 
    
    % 更新超参数
    gamma_new = abs(mu).^2 + gamma - gamma.^2 .* real(diag_term);
    
    % 【修复3】截断微小噪声 (剪枝)，强制稀疏度并防止数值崩塌
    gamma_new(gamma_new < gamma_tol) = gamma_tol;
    
    % 检查收敛
    if norm(gamma_new - gamma)/norm(gamma) < 1e-3
        gamma = gamma_new;
        break;
    end
    gamma = gamma_new;
end

% 【修复4】寻峰策略优化：设置最小距离，防止 SBL 能量泄露导致的假峰
[~, locs_sbl] = findpeaks(gamma, 'SortStr', 'descend', 'MinPeakDistance', 5);
if length(locs_sbl) < L_paths
    [~, locs_sbl] = sort(gamma, 'descend'); % 容错
end
idx_sbl = locs_sbl(1:L_paths);

est_delays_sbl = tau_grid(idx_sbl)';
% SBL 的均值 mu 已经直接包含了幅度信息，可以直接使用
est_amps_sbl = mu(idx_sbl); 
residual_sbl = y_rx - A_dict(:, idx_sbl) * est_amps_sbl;

%% 7. 算法 F: NOMP (Gridless OMP) — OMP粗搜 + 局部细网格精化连续时延
% Step 1: OMP粗搜（复用字典，获取粗时延和初始幅度）
residual_nomp_temp = y_rx;
idx_nomp_coarse = zeros(L_paths, 1);
tau_nomp_coarse = zeros(L_paths, 1);
for l = 1:L_paths
    corr_nomp = abs(A_dict' * residual_nomp_temp);
    [~, best_idx] = max(corr_nomp);
    idx_nomp_coarse(l) = best_idx;
    tau_nomp_coarse(l) = tau_grid(best_idx);
    A_sub = A_dict(:, idx_nomp_coarse(1:l));
    amps_tmp = A_sub \ y_rx;
    residual_nomp_temp = y_rx - A_sub * amps_tmp;
end

% Step 2: 迭代局部细网格精化 + 干扰消除 (类SAGE机制, 3轮外循环)
delta_fine = 0.005;          % 精化步长 (与论文Stage 2一致)
search_halfwin = 1.0;        % 搜索半径 ±1 样本
outer_iters  = 3;            % 外循环 (精化 → 更新幅度 → 更新残差)
tau_nomp_refined = tau_nomp_coarse;
amps_nomp = amps_tmp;

% 初始化残差 (减去所有路径的当前估计)
residual_nomp = y_rx;
for ll = 1:L_paths
    s_l = fractional_delay_shift(s_tx_base, tau_nomp_refined(ll), N_rx);
    residual_nomp = residual_nomp - amps_nomp(ll) * s_l;
end

for outer = 1:outer_iters
    for l = 1:L_paths
        % --- E-step: 重建当前路径的隐藏信号 ---
        s_old = amps_nomp(l) * fractional_delay_shift(s_tx_base, tau_nomp_refined(l), N_rx);
        x_l = residual_nomp + s_old;

        % --- 局部细网格搜索 ---
        tau_min = tau_nomp_refined(l) - search_halfwin;
        tau_max = tau_nomp_refined(l) + search_halfwin;
        tau_candidates = tau_min : delta_fine : tau_max;

        best_corr = -inf;
        best_tau = tau_nomp_refined(l);
        for tc = 1:length(tau_candidates)
            tau_test = tau_candidates(tc);
            s_test = fractional_delay_shift(s_tx_base, tau_test, N_rx);
            corr_test = abs(s_test' * x_l);
            if corr_test > best_corr
                best_corr = corr_test;
                best_tau = tau_test;
            end
        end
        tau_nomp_refined(l) = best_tau;

        % --- 更新幅度与残差 ---
        s_new = fractional_delay_shift(s_tx_base, best_tau, N_rx);
        amps_nomp(l) = (s_new' * x_l) / (s_tx_base' * s_tx_base);
        residual_nomp = x_l - amps_nomp(l) * s_new;
    end
end

% Step 3: 最终联合最小二乘重估幅度
A_nomp_refined = zeros(N_rx, L_paths);
for l = 1:L_paths
    A_nomp_refined(:, l) = fractional_delay_shift(s_tx_base, tau_nomp_refined(l), N_rx);
end
amps_nomp_refined = A_nomp_refined \ y_rx;
est_delays_nomp = tau_nomp_refined;
est_amps_nomp = amps_nomp_refined;
residual_nomp = y_rx - A_nomp_refined * amps_nomp_refined;

%% 8. 后处理：路径自动匹配 (解决乱序问题)
match_delays = zeros(L_paths, 6); match_amps = zeros(L_paths, 6); % [Int, Frac, MF, OMP, SBL, NOMP]

for i = 1:L_paths
    [~, id1] = min(abs(est_delays_int - true_delays(i))); match_delays(i,1) = est_delays_int(id1); match_amps(i,1) = est_amps_int(id1);
    [~, id2] = min(abs(est_delays_frac - true_delays(i))); match_delays(i,2) = est_delays_frac(id2); match_amps(i,2) = est_amps_frac(id2);
    [~, id3] = min(abs(est_delays_mf - true_delays(i))); match_delays(i,3) = est_delays_mf(id3); match_amps(i,3) = est_amps_mf(id3);
    [~, id4] = min(abs(est_delays_omp - true_delays(i))); match_delays(i,4) = est_delays_omp(id4); match_amps(i,4) = est_amps_omp(id4);
    [~, id5] = min(abs(est_delays_sbl - true_delays(i))); match_delays(i,5) = est_delays_sbl(id5); match_amps(i,5) = est_amps_sbl(id5);
    [~, id6] = min(abs(est_delays_nomp - true_delays(i))); match_delays(i,6) = est_delays_nomp(id6); match_amps(i,6) = est_amps_nomp(id6);
end

%% 9. 控制台输出全维度对比表
fprintf('\n=================================================================================\n');
fprintf(' 多径时延估计算法全面性能对比 (基于 %d 次采样, 信噪比相关噪音标准差=%.2f)\n', N_rx, noise_std);
fprintf('=================================================================================\n');

for i = 1:L_paths
    t_tau = true_delays(i); t_mag = abs(true_amps(i)); t_phase = angle(true_amps(i));

    fprintf('【多径 %d】 理论 CRB(时延) = %.5f\n', i, std_CRB(i));
    fprintf('  [真实值]     时延: %6.2f | 幅度: %5.3f | 相位: %6.3f rad\n', t_tau, t_mag, t_phase);

    alg_names = {'整数 SAGE', '小数 SAGE', 'MF (网格1.0)', 'OMP(网格1.0)', 'SBL(网格1.0)', 'NOMP'};
    for j = 1:6
        tau = match_delays(i,j);   e_tau = abs(tau - t_tau);
        mag = abs(match_amps(i,j)); e_mag = abs(mag - t_mag);
        pha = angle(match_amps(i,j));e_pha = abs(wrapToPi(pha - t_phase));
        fprintf('  [%-10s] 时延: %6.4f (err:%6.4f) | 幅度: %5.3f (err:%5.3f) | 相位: %6.3f (err:%5.3f)\n', ...
            alg_names{j}, tau, e_tau, mag, e_mag, pha, e_pha);
    end
    fprintf('---------------------------------------------------------------------------------\n');
end

fprintf('最终残差能量对比:\n');
fprintf('  整数 SAGE : %.4f | 小数 SAGE : %.4f | MF : %.4f | OMP : %.4f | SBL : %.4f | NOMP : %.4f\n', ...
    norm(residual_int)^2, norm(residual_frac)^2, norm(residual_mf)^2, norm(residual_omp)^2, norm(residual_sbl)^2, norm(residual_nomp)^2);
fprintf('=================================================================================\n');

%% 10. 学术级对比绘图

% 图1：残差信号对比 (揭示”路径分裂”现象的核心图)
figure('Color', 'w', 'Position', [100 100 1200 400]);
plot(abs(residual_int), 'k-', 'LineWidth', 1); hold on;
plot(abs(residual_omp), 'm--', 'LineWidth', 1);
plot(abs(residual_sbl), 'c-.', 'LineWidth', 1);
plot(abs(residual_frac), 'b-', 'LineWidth', 1.5);
plot(abs(residual_nomp), '-', 'Color', [0.85 0.33 0.1], 'LineWidth', 1.5);
title('最终残差信号 (Residual Error) 对比', 'FontSize', 12);
xlabel('时间 (采样点)', 'FontSize', 11); ylabel('残差绝对值', 'FontSize', 11);
legend('整数 SAGE (基线)', 'OMP', 'SBL', '小数 SAGE (本文)', 'NOMP', 'Location', 'best');
grid on; xlim([30, 250]);

% 图2：冲激响应提取大图
fig = figure('Color', 'w', 'Position', [100 200 1300 650]); 
pos_main = [0.1, 0.15, 0.85, 0.8]; 
pos_sub1 = [0.23, 0.60, 0.25, 0.30]; % 放大子图1 
pos_sub2 = [0.23, 0.20, 0.25, 0.30]; % 放大子图1 
pos_sub3 = [0.55, 0.20, 0.25, 0.30]; % 放大子图2 

% 第1步：绘制主图
ax_main = axes('Position', pos_main);
hold(ax_main, 'on'); 
draw_all_stems(ax_main, true_delays, true_amps, match_delays, match_amps);
xlabel(ax_main, 'Delay (Sample)', 'FontSize', 11); ylabel(ax_main, 'Amptitude |\alpha|', 'FontSize', 11);
legend(ax_main, 'True (Continuous)', 'Int SAGE', 'Frac SAGE', 'MF (Grid)', 'OMP (Grid)', 'SBL (Grid)', 'NOMP', ...
    'Location', 'northeast', 'FontSize', 9, 'Box','off');
grid(ax_main, 'on'); xlim(ax_main, [40, 220]); ylim(ax_main, [0, 1.2]);
set(ax_main, 'FontName', 'Times New Roman', 'FontSize', 10);

% 第2/3步：放大区域 1 (48~68)
box_xlim1 = [49, 52]; box_ylim1 = [0, 1.1]; 
rectangle(ax_main, 'Position', [box_xlim1(1), box_ylim1(1), diff(box_xlim1), diff(box_ylim1)], ...
    'LineStyle', '--', 'LineWidth', 1.2, 'EdgeColor', [0.3 0.3 0.3]);
ax_sub1 = axes('Position', pos_sub1); hold(ax_sub1, 'on'); box(ax_sub1, 'on'); grid(ax_sub1, 'on'); 
draw_all_stems(ax_sub1, true_delays, true_amps, match_delays, match_amps);
xlim(ax_sub1, box_xlim1); ylim(ax_sub1, box_ylim1);

box_xlim2 = [59, 62]; box_ylim2 = [0, 0.6]; 
rectangle(ax_main, 'Position', [box_xlim2(1), box_ylim2(1), diff(box_xlim2), diff(box_ylim2)], ...
    'LineStyle', '--', 'LineWidth', 1.2, 'EdgeColor', [0.3 0.3 0.3]);
ax_sub2 = axes('Position', pos_sub2); hold(ax_sub2, 'on'); box(ax_sub2, 'on'); grid(ax_sub2, 'on'); 
draw_all_stems(ax_sub2, true_delays, true_amps, match_delays, match_amps);
xlim(ax_sub2, box_xlim2); ylim(ax_sub2, box_ylim2);

% 第4/5步：放大区域 2 (198~202)
box_xlim3 = [199.5, 201.5]; box_ylim3 = [0, 0.9]; 
rectangle(ax_main, 'Position', [box_xlim3(1), box_ylim3(1), diff(box_xlim3), diff(box_ylim3)], ...
    'LineStyle', '--', 'LineWidth', 1.2, 'EdgeColor', [0 0 1]);
ax_sub3 = axes('Position', pos_sub3); hold(ax_sub3, 'on'); box(ax_sub3, 'on'); grid(ax_sub3, 'on'); 
draw_all_stems(ax_sub3, true_delays, true_amps, match_delays, match_amps);
xlim(ax_sub3, box_xlim3); ylim(ax_sub3, box_ylim3);

% 保存图2（冲激响应）为论文用图
saveas(fig, 'results/fig_09_channel_stem_mat.png');

%% 11. RMSE 汇总对比图 (替换论文 fig_08)
alg_names_plot = {'Int SAGE', 'Frac SAGE', 'MF', 'OMP', 'SBL', 'NOMP'};
err_matrix = abs(match_delays - true_delays');           % L_paths × 6
rmse_per_alg = sqrt(mean(err_matrix.^2, 1));            % 1 × 6, 综合三条路径的RMSE

figure('Color', 'w', 'Position', [100 100 1100 550]);

% --- 子图1: 各算法综合RMSE柱状图 ---
subplot(1,2,1);
colors_bar = {[0.3 0.3 0.3], [0 0.45 0.74], [0.47 0.67 0.19], [0.64 0.08 0.18], [0 0.75 0.75], [0.85 0.33 0.1]};
b = bar(rmse_per_alg, 'FaceColor', 'flat', 'EdgeColor', 'w', 'LineWidth', 0.8);
for k = 1:6, b.CData(k,:) = colors_bar{k}; end
set(gca, 'XTickLabel', alg_names_plot, 'FontSize', 10, 'FontName', 'Times New Roman');
xtickangle(25);
ylabel('Delay RMSE (samples)', 'FontSize', 11);
title(sprintf('Overall RMSE (3 paths, SNR~26 dB)'), 'FontWeight', 'bold', 'FontSize', 11);
grid on; box on;

% 标注: 柱顶 = RMSE值, 柱内 = 相对Int SAGE的提升百分比
rmse_baseline = rmse_per_alg(1);  % Int SAGE作为基线
for k = 1:6
    % 柱顶标注RMSE绝对值
    text(k, rmse_per_alg(k)*1.15, sprintf('%.3f', rmse_per_alg(k)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold', ...
        'Color', colors_bar{k});
    % 柱内标注改善百分比 (相对Int SAGE)
    impr_pct = (rmse_baseline - rmse_per_alg(k)) / rmse_baseline * 100;
    if k == 1
        label_str = '(baseline)';
        txt_color = 'w';
    elseif impr_pct > 0
        label_str = sprintf('%.1f%%', impr_pct);
        txt_color = 'w';
    else
        label_str = sprintf('%.1f%%', impr_pct);
        txt_color = 'y';
    end
    text(k, rmse_per_alg(k)*0.45, label_str, ...
        'HorizontalAlignment', 'center', 'FontSize', 8.5, 'FontWeight', 'bold', ...
        'Color', txt_color);
end
% CRB参考线
crb_avg = mean(std_CRB);
hold on; yline(crb_avg, 'r--', 'CRB', 'LineWidth', 1.5);

% --- 子图2: 逐径时延误差对比 ---
subplot(1,2,2);
alg_colors = lines(6);
for alg = 1:6
    semilogy(1:L_paths, err_matrix(:,alg)', 'o-', 'LineWidth', 1.8, 'MarkerSize', 9, ...
        'Color', colors_bar{alg}, 'MarkerFaceColor', colors_bar{alg});
    hold on;
end
% CRB per path
semilogy(1:L_paths, std_CRB, 'r--', 'LineWidth', 2, 'Marker', 's', 'MarkerSize', 10);
set(gca, 'XTick', 1:L_paths, 'XTickLabel', {'Path 1 (50.43)', 'Path 2 (60.66)', 'Path 3 (200.51)'}, ...
    'FontSize', 9, 'FontName', 'Times New Roman');
xtickangle(12);
ylabel('Delay Error (samples)', 'FontSize', 11);
legend([alg_names_plot, {'CRB'}], 'Location', 'northeastoutside', 'FontSize', 8);
title('Per-Path Delay Estimation Error', 'FontWeight', 'bold', 'FontSize', 11);
grid on; box on;

saveas(gcf, 'results/fig_08_rmse_vs_snr_mat.png');

fprintf('\n>>> 图片已保存至 results/ 目录\n');
function draw_all_stems(ax, t_tau, t_amp, m_tau, m_amp)
    lw = 1.5;
    % 为了视觉可辨识度，在X轴上施加微小偏移（仅用于绘图，不改变真实数据）
    % 真实信道
    stem(ax, t_tau, abs(t_amp), 'r', 'LineWidth', 2, 'Marker', 'o');
    % 算法 1-6 (Int, Frac, MF, OMP, SBL, NOMP)
    stem(ax, m_tau(:,1) - 0.15, abs(m_amp(:,1)), 'k', 'LineWidth', lw, 'Marker', 's');
    stem(ax, m_tau(:,2)       , abs(m_amp(:,2)), 'b-', 'LineWidth', 2, 'Marker', '^');
    stem(ax, m_tau(:,3) + 0.15, abs(m_amp(:,3)), 'g--', 'LineWidth', lw, 'Marker', '*');
    stem(ax, m_tau(:,4) + 0.30, abs(m_amp(:,4)), 'm--', 'LineWidth', lw, 'Marker', 'd');
    stem(ax, m_tau(:,5) - 0.30, abs(m_amp(:,5)), 'c-.', 'LineWidth', lw, 'Marker', 'x');
    stem(ax, m_tau(:,6) + 0.45, abs(m_amp(:,6)), 'Color', [0.85 0.33 0.1], 'LineWidth', 1.8, 'Marker', 'p', 'MarkerSize', 8);
end

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