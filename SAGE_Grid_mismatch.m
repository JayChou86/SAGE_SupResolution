% =========================================================================
% 学术级实验：栅格失配与亚采样 SAGE 性能敏感度扫描
% 目的：探究不同采样率下，整数网格误差对信道估计的影响
% =========================================================================
clear; clc; close all;

load('data475.mat'); 
y_raw_orig = sig_rev_m(:); 
s_tx_orig  = sig_bd_m(:);  

% 定义要扫描的降采样倍数 M = 1,2,...,16 (覆盖过采样→欠采样全区间)
down_ratios = 1:16;

num_tests = length(down_ratios);

mse_int_list = zeros(num_tests, 1);
mse_frac_list = zeros(num_tests, 1);
improvement_list = zeros(num_tests, 1);

fprintf('开始进行采样率敏感度扫描实验...\n');

for test_idx = 1:num_tests
    ratio = down_ratios(test_idx);
    
    % 将小数倍数转化为有理分数 P/Q (例如 3.45 -> resample(x, 100, 345))
    [num, den] = rat(1 / ratio); 
    
    % 【核心修正】：使用严格的带通重采样函数，避免索引截断带来的 Jitter
    y_raw = resample(y_raw_orig, num, den);
    s_tx  = resample(s_tx_orig, num, den);
    
    N = length(y_raw);    
    Es = sum(abs(s_tx).^2); 

    S_f = fft(s_tx);
    Y_f = fft(y_raw);

    if mod(N,2) == 0
        f_vec = [0:N/2-1, -N/2:-1]' / N;
    else
        f_vec = [0:(N-1)/2, -(N-1)/2:-1]' / N;
    end
    
    % --------- 以下为 SAGE 算法核心 (与之前完全一致，为节省版面省略部分注释) ---------
    K = 8; num_iter = 5;
    tau_int = zeros(K,1); alpha_int = zeros(K,1);
    tau_frac = zeros(K,1); alpha_frac = zeros(K,1);
    
    % 1. Integer SAGE
    y_res_f = Y_f; alpha_k_f = zeros(N, K); 
    for iter = 1:num_iter
        for k = 1:K
            x_k_f = y_res_f + alpha_k_f(:, k) .* S_f;
            R_k = ifft(x_k_f .* conj(S_f)) / Es;
            [~, idx] = max(abs(R_k));
            tau_int(k) = idx - 1; alpha_int(k) = R_k(idx);
            alpha_k_f(:, k) = alpha_int(k) .* exp(-1i * 2 * pi * f_vec * tau_int(k));
            y_res_f = x_k_f - alpha_k_f(:, k) .* S_f;
        end
    end
    
    % 2. Fractional SAGE
    y_res_f_frac = Y_f; alpha_k_f_frac = zeros(N, K);
    for iter = 1:num_iter
        for k = 1:K
            x_k_f = y_res_f_frac + alpha_k_f_frac(:, k) .* S_f;
            R_k = ifft(x_k_f .* conj(S_f)) / Es;
            [~, idx] = max(abs(R_k));
            if idx > 1 && idx < N
                y_m1 = abs(R_k(idx - 1)); y_0 = abs(R_k(idx)); y_p1 = abs(R_k(idx + 1));
                delta = 0.5 * (y_m1 - y_p1) / (y_m1 - 2*y_0 + y_p1);
                delta = max(min(delta, 0.5), -0.5); 
            else
                delta = 0;
            end
            tau_curr = (idx - 1) + delta; tau_frac(k) = tau_curr;
            phase_shift = exp(1i * 2 * pi * f_vec * tau_curr);
            exact_amp = sum((x_k_f .* conj(S_f)) .* phase_shift) / (N * Es);
            alpha_frac(k) = exact_amp;
            alpha_k_f_frac(:, k) = alpha_frac(k) .* exp(-1i * 2 * pi * f_vec * tau_frac(k));
            y_res_f_frac = x_k_f - alpha_k_f_frac(:, k) .* S_f;
        end
    end
    
    % 3. 均衡与 MSE 计算
    H_int = zeros(N,1); H_frac = zeros(N,1);
    for k = 1:K
        H_int  = H_int  + alpha_int(k)  .* exp(-1i * 2 * pi * f_vec * tau_int(k));
        H_frac = H_frac + alpha_frac(k) .* exp(-1i * 2 * pi * f_vec * tau_frac(k));
    end
    noise_var = 0.05 * max(abs(H_frac))^2; 
    
    S_eq_int_f  = Y_f .* conj(H_int)  ./ (abs(H_int).^2  + noise_var);
    S_eq_frac_f = Y_f .* conj(H_frac) ./ (abs(H_frac).^2 + noise_var);
    s_eq_int  = ifft(S_eq_int_f);
    s_eq_frac = ifft(S_eq_frac_f);
    
    valid_idx = 1000:(N-1000); 
    mse_int_list(test_idx)  = mean(abs(s_eq_int(valid_idx) - s_tx(valid_idx)).^2);
    mse_frac_list(test_idx) = mean(abs(s_eq_frac(valid_idx) - s_tx(valid_idx)).^2);
    
    improvement_list(test_idx) = (mse_int_list(test_idx) - mse_frac_list(test_idx)) / mse_int_list(test_idx) * 100;
    
    fprintf('降采样倍数 %.2f | Int MSE: %.4e | Frac MSE: %.4e | 提升: %5.2f %%\n', ...
        ratio, mse_int_list(test_idx), mse_frac_list(test_idx), improvement_list(test_idx));
end

%% 绘制学术级图表：栅格失配敏感度 (匹配Python fig_12风格)
figure('Name', 'Grid Mismatch Sensitivity', 'Color', 'w', 'Position', [50, 150, 1200, 550]);

% --- 左轴: MSE (semilogy) ---
yyaxis left;
semilogy(down_ratios, mse_int_list, 'b-s', 'LineWidth', 2, 'MarkerSize', 8, ...
    'MarkerFaceColor', 'b'); hold on;
semilogy(down_ratios, mse_frac_list, 'r-o', 'LineWidth', 2, 'MarkerSize', 8, ...
    'MarkerFaceColor', 'r');
ylabel('Equalization MSE', 'FontSize', 12, 'FontWeight', 'bold');
% 左轴留余量
all_mse = [mse_int_list; mse_frac_list];
ylim([min(all_mse)*0.9, max(all_mse)*1.1]);
set(gca, 'YColor', 'k', 'FontSize', 10, 'FontName', 'Times New Roman');

% --- 右轴: Improvement bar ---
yyaxis right;
bar_colors = zeros(length(improvement_list), 3);
for i = 1:length(improvement_list)
    if improvement_list(i) < 0
        bar_colors(i,:) = [0.55 0 0];     % darkred
    else
        bar_colors(i,:) = [0 0.39 0];     % darkgreen
    end
end
b = bar(down_ratios, improvement_list, 0.5, 'FaceColor', 'flat', ...
    'EdgeColor', [0.33 0.33 0.33], 'LineWidth', 0.8, 'FaceAlpha', 0.5);
b.CData = bar_colors;
ylabel('Improvement (%)', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'YColor', 'k', 'FontSize', 10, 'FontName', 'Times New Roman');

% --- 柱底标注百分比 ---
yyaxis right;  % 确保在右轴操作
y_range = max(abs(improvement_list));
ylim([min(improvement_list) - y_range*0.4, max(improvement_list) + y_range*0.15]);

for i = 1:length(improvement_list)
    v = improvement_list(i);
    y_pos = 0 - y_range * 0.05;           % 统一在0轴下方（柱底之下）
    if v >= 0
        txt = sprintf('+%.1f%%', v);
    else
        txt = sprintf('-%.1f%%', abs(v));
    end
    text(down_ratios(i), y_pos, txt, ...
        'HorizontalAlignment', 'center', 'FontSize', 8.5, 'FontWeight', 'bold', ...
        'Color', bar_colors(i,:), 'FontName', 'Times New Roman');
end

% --- 参考线 ---
yline(0, 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8);

xlabel('Decimation Factor $M$', 'FontSize', 12, 'FontWeight', 'bold', ...
    'FontName', 'Times New Roman', 'Interpreter', 'latex');
title('Grid Mismatch Sensitivity — Sea Trial ($K=8$)', 'FontWeight', 'bold', ...
    'FontSize', 13, 'FontName', 'Times New Roman');

legend('Integer SAGE', 'Fractional SAGE', 'Location', 'northwest', ...
    'FontSize', 10, 'FontName', 'Times New Roman', 'Box', 'off');
set(gca, 'XTick', down_ratios);
grid on; box on;

% --- 保存图片 ---
saveas(gcf, 'results/fig_12_seatrial_mat.png');
fprintf('\n>>> 图片已保存至 results/fig_12_seatrial_mat.png\n');