% =========================================================================
% SAGE 算法性能对比评估框架：整数搜索(基线) vs 亚采样超分辨率(本文提出)
% 包含：严格的路径自动匹配、时延/幅度/相位全参数对比、残差能量对比
% =========================================================================
clear; clc; close all;

%% 1. 生成带小数时延的仿真信号 (物理真实信道)
fs = 10000; 
N_seq = 511; 
rng(42); % 固定随机种子
s_tx_base = sign(randn(N_seq, 1)) + 1j * sign(randn(N_seq, 1));
s_tx_base = s_tx_base / norm(s_tx_base);

% 设置带小数点的真实时延，以及复数幅度(包含幅值和相位)
true_delays = [50.4, 65.7, 200.2]; 
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

%% 2. 算法 A: 传统 SAGE 算法 (仅整数搜索 - 无亚采样)
max_iter = 10;
est_delays_int = zeros(L_paths, 1); 
est_amps_int   = zeros(L_paths, 1); 
residual_int   = y_rx;

for iter = 1:max_iter
    for l = 1:L_paths
        % E-step
        if iter > 1
            s_old = est_amps_int(l) * fractional_delay_shift(s_tx_base, est_delays_int(l), N_rx);
            x_l = residual_int + s_old;
        else
            x_l = residual_int;
        end
        
        % M-step (仅整数点搜索)
        search_len = N_rx - N_seq;
        corr_vals = zeros(search_len, 1);
        for tau = 1:search_len
            corr_vals(tau) = abs(s_tx_base' * x_l(tau : tau + N_seq - 1));
        end
        [~, k_opt] = max(corr_vals);
        
        tau_int = k_opt; % 强制为整数
        
        % 提取复幅度
        s_int_temp = fractional_delay_shift(s_tx_base, tau_int, N_rx);
        amp_int = (s_int_temp' * x_l) / (s_tx_base' * s_tx_base);
        
        % 更新
        est_delays_int(l) = tau_int;
        est_amps_int(l)   = amp_int;
        residual_int = x_l - amp_int * s_int_temp;
    end
end

%% 3. 算法 B: 超分辨率 SAGE 算法 (带抛物线插值亚采样)
est_delays_frac = zeros(L_paths, 1); 
est_amps_frac   = zeros(L_paths, 1); 
residual_frac   = y_rx;

for iter = 1:max_iter
    for l = 1:L_paths
        % E-step
        if iter > 1
            s_old = est_amps_frac(l) * fractional_delay_shift(s_tx_base, est_delays_frac(l), N_rx);
            x_l = residual_frac + s_old;
        else
            x_l = residual_frac;
        end
        
        % M-step (先整数搜索，再抛物线插值)
        search_len = N_rx - N_seq;
        corr_vals = zeros(search_len, 1);
        for tau = 1:search_len
            corr_vals(tau) = abs(s_tx_base' * x_l(tau : tau + N_seq - 1));
        end
        [~, k_opt] = max(corr_vals);
        
        % --- 核心：抛物线亚采样插值 ---
        if k_opt > 1 && k_opt < search_len
            y_m1 = corr_vals(k_opt - 1);
            y_0  = corr_vals(k_opt);
            y_p1 = corr_vals(k_opt + 1);
            delta = 0.5 * (y_m1 - y_p1) / (y_m1 - 2*y_0 + y_p1);
            tau_frac = k_opt + delta;
        else
            tau_frac = k_opt;
        end
        
        % 提取复幅度 (基于带小数的时延重构波形)
        s_frac_temp = fractional_delay_shift(s_tx_base, tau_frac, N_rx);
        amp_frac = (s_frac_temp' * x_l) / (s_tx_base' * s_tx_base);
        
        % 更新
        est_delays_frac(l) = tau_frac;
        est_amps_frac(l)   = amp_frac;
        residual_frac = x_l - amp_frac * s_frac_temp;
    end
end

%% 4. 后处理：路径自动匹配 (解决乱序问题)
% 匹配逻辑：对于每一条真实多径，在估计结果中寻找距离（时延差）最近的路径
match_int_delays = zeros(L_paths, 1); match_int_amps = zeros(L_paths, 1);
match_frac_delays = zeros(L_paths, 1); match_frac_amps = zeros(L_paths, 1);

for i = 1:L_paths
    % 匹配整数算法的结果
    [~, idx_int] = min(abs(est_delays_int - true_delays(i)));
    match_int_delays(i) = est_delays_int(idx_int);
    match_int_amps(i)   = est_amps_int(idx_int);
    
    % 匹配分数算法的结果
    [~, idx_frac] = min(abs(est_delays_frac - true_delays(i)));
    match_frac_delays(i) = est_delays_frac(idx_frac);
    match_frac_amps(i)   = est_amps_frac(idx_frac);
end

%% 5. 控制台输出全维度对比表
fprintf('=================================================================================\n');
fprintf(' SAGE 算法性能对比评估 (真实值 vs 无亚采样 vs 有亚采样)\n');
fprintf('=================================================================================\n');

for i = 1:L_paths
    % 真实值
    t_tau   = true_delays(i);
    t_mag   = abs(true_amps(i));
    t_phase = angle(true_amps(i));
    
    % 算法A (整数) 提取值及误差
    i_tau   = match_int_delays(i);     e_i_tau = abs(i_tau - t_tau);
    i_mag   = abs(match_int_amps(i));  e_i_mag = abs(i_mag - t_mag);
    i_phase = angle(match_int_amps(i));e_i_pha = abs(wrapToPi(i_phase - t_phase));
    
    % 算法B (分数) 提取值及误差
    f_tau   = match_frac_delays(i);      e_f_tau = abs(f_tau - t_tau);
    f_mag   = abs(match_frac_amps(i));   e_f_mag = abs(f_mag - t_mag);
    f_phase = angle(match_frac_amps(i)); e_f_pha = abs(wrapToPi(f_phase - t_phase));
    
    fprintf('【多径 %d】\n', i);
    fprintf('  [真实值]   时延: %6.2f | 幅度: %5.3f | 相位: %6.3f rad\n', t_tau, t_mag, t_phase);
    fprintf('  [无亚采样] 时延: %6.2f (err:%5.2f) | 幅度: %5.3f (err:%5.3f) | 相位: %6.3f (err:%5.3f)\n', ...
        i_tau, e_i_tau, i_mag, e_i_mag, i_phase, e_i_pha);
    fprintf('  [有亚采样] 时延: %6.4f (err:%5.4f) | 幅度: %5.3f (err:%5.3f) | 相位: %6.3f (err:%5.3f)\n', ...
        f_tau, e_f_tau, f_mag, e_f_mag, f_phase, e_f_pha);
    fprintf('---------------------------------------------------------------------------------\n');
end

fprintf('最终残差能量对比:\n');
fprintf('  无亚采样残差能量: %.4f\n', norm(residual_int)^2);
fprintf('  有亚采样残差能量: %.4f\n', norm(residual_frac)^2);
fprintf('=================================================================================\n');

%% 6. 学术级对比绘图
figure('Color', 'w', 'Position', [100 100 1200 500]);

% 图1：信道冲激响应包络对比
% subplot(1,2,1);
stem(true_delays, abs(true_amps), 'r', 'LineWidth', 2, 'Marker', 'o'); hold on;
stem(match_int_delays, abs(match_int_amps), 'k', 'LineWidth', 1.5, 'Marker', 's');
stem(match_frac_delays, abs(match_frac_amps), 'b--', 'LineWidth', 2, 'Marker', '^');
title('多径时延与幅度估计结果对比', 'FontSize', 12);
xlabel('时延 (采样点)', 'FontSize', 11); ylabel('复数幅度包络 |\alpha|', 'FontSize', 11);
legend('真实信道 (连续时延)', '无亚采样 SAGE (受限于网格)', '有亚采样 SAGE (突破网格)', 'Location', 'best');
grid on; xlim([30, 220]);


% 图2：残差信号对比 (揭示“路径分裂”现象的核心图)
figure('Color', 'w', 'Position', [100 100 1200 500]);
% subplot(1,2,2);
plot(abs(residual_int), 'k', 'LineWidth', 1); hold on;
plot(abs(residual_frac), 'b', 'LineWidth', 1.5);
title('最终残差信号 (Residual Error) 对比', 'FontSize', 12);
xlabel('时间 (采样点)', 'FontSize', 11); ylabel('残差绝对值', 'FontSize', 11);
legend(sprintf('无亚采样残差 (能量=%.1f)', norm(residual_int)^2), ...
       sprintf('有亚采样残差 (能量=%.1f)', norm(residual_frac)^2), 'Location', 'best');
grid on; xlim([30, 250]);


figure('Color', 'w', 'Position', [100 100 1200 650]); % 稍微增高一点图形

% --- 第1步：绘制主图（全景图） ---
% 我们使用 axes 来更精确地控制位置，让主图在左侧，留出右侧空间给放大图
ax_main = axes('Position', [0.08, 0.15, 0.5, 0.75]); % X, Y, Width, Height

% 调用自定义函数进行绘制（这样下面的子图也可以重用这段代码）
stem_custom_plot(ax_main, true_delays, true_amps, match_int_delays, match_int_amps, match_frac_delays, match_frac_amps);

% 设置主图的轴、标记和标题
title(ax_main, '多径时延与幅度估计结果对比（全景图）', 'FontSize', 12);
xlabel(ax_main, '时延 (采样点)', 'FontSize', 11); ylabel(ax_main, '复数幅度包络 |\alpha|', 'FontSize', 11);
legend(ax_main, '真实信道 (连续时延)', '无亚采样 SAGE (受限于网格)', '有亚采样 SAGE (突破网格)', 'Location', 'best','FontSize',9);
grid(ax_main, 'on'); 
xlim(ax_main, [30, 220]); % 设置全景图的 X轴范围
ylim(ax_main, [0,1.2])

% --- 第2步：绘制放大子图1（放大 x 轴 48-68） ---
% 创建新的坐标轴，放置在主图右侧的特定位置
ax_sub1 = axes('Position', [0.65, 0.55, 0.3, 0.35]); % 调整位置和大小
box(ax_sub1, 'on'); % 加上小盒子
grid(ax_sub1,'on');

% 在新轴上再次绘制
stem_custom_plot(ax_sub1, true_delays, true_amps, match_int_delays, match_int_amps, match_frac_delays, match_frac_amps);
title(ax_sub1, '放大子图 1 (48~68 采样点)', 'FontSize', 11);
xlabel(ax_sub1, '时延 (采样点)', 'FontSize', 10); ylabel(ax_sub1, '幅度包络 |\alpha|', 'FontSize', 10);

% 设置子图1的特定缩放范围
xlim(ax_sub1, [48, 68]); 
ylim(ax_sub1, [0, 1.1]); % 适当调整Y轴范围，突出细节

% --- 第3步：绘制放大子图2（放大 x 轴 198-202） ---
% 创建第二个新的坐标轴，放置在子图1下方
ax_sub2 = axes('Position', [0.65, 0.1, 0.3, 0.35]); % 调整位置
box(ax_sub2, 'on'); % 加上小盒子
grid(ax_sub2,'on');

% 在新轴上再次绘制
stem_custom_plot(ax_sub2, true_delays, true_amps, match_int_delays, match_int_amps, match_frac_delays, match_frac_amps);
title(ax_sub2, '放大子图 2 (198~202 采样点)', 'FontSize', 11);
xlabel(ax_sub2, '时延 (采样点)', 'FontSize', 10); ylabel(ax_sub2, '幅度包络 |\alpha|', 'FontSize', 10);

% 设置子图2的特定缩放范围
xlim(ax_sub2, [199.5, 201.5]);
ylim(ax_sub2, [0, 1]); % 适当调整Y轴范围


% --- 定义绘图的局部辅助函数 ---
function stem_custom_plot(ax, d1, a1, d2, a2, d3, a3)
% 这里的 ax 参数指定要在哪个轴对象上进行绘制。
% 关键是要使用 'Parent', ax，以便准确控制。
stem(ax, d1, abs(a1), 'r', 'LineWidth', 2, 'Marker', 'o','Parent',ax); 
hold(ax, 'on');
stem(ax, d2, abs(a2), 'k', 'LineWidth', 1.5, 'Marker', 's','Parent',ax);
stem(ax, d3, abs(a3), 'b--', 'LineWidth', 2, 'Marker', '^','Parent',ax);
end

%% --- 辅助函数：严格基于 FFT 的分数延迟 (修复了频谱泄露) ---
function s_shifted = fractional_delay_shift(s_base, delay, N_total)
    N_orig = length(s_base);
    S_freq = fft(s_base);
    
    % 构造频域相移算子 (使用对称频率向量以保证无失真重构)
    k = [0:ceil(N_orig/2)-1, -floor(N_orig/2):-1]';
    phase_shift = exp(-1j * 2 * pi * k * (delay - floor(delay)) / N_orig);
    s_temp = ifft(S_freq .* phase_shift);
    
    s_shifted = zeros(N_total, 1);
    start_idx = floor(delay);
    if start_idx >= 1 && start_idx + N_orig - 1 <= N_total
        s_shifted(start_idx : start_idx + N_orig - 1) = s_temp;
    end
end