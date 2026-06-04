% =========================================================================
% 基于复基带原始信号的高分辨率 SAGE 多径提取算法 (正统理论版)
% 
% =========================================================================
clear; clc; close all;

%% 1. 生成仿真用的“原始复基带观测信号” (模拟海试真实接收)
% 【这部分在实际处理时，请替换为你海试录下的、经过下变频的复数基带数据 y_rx】

fs = 10000;          % 采样率
N_seq = 511;         % M序列长度 (这里用随机伪随机码代替)
rng(42);             % 固定随机种子，方便复现
s_tx = sign(randn(N_seq, 1)) + 1j * sign(randn(N_seq, 1)); % 生成 QPSK 形式的复数导频/M序列
s_tx = s_tx / norm(s_tx); % 归一化发射波形能量为 1

% 设定真实信道的 3 条多径 (时延设为整数个采样点)
% 注意：第2条径很弱，且离第1条径很近，很容易被第一条径的旁瓣掩盖！
true_delays = [50, 60, 200];               % 真实时延 (采样点索引)
true_amps   = [1.0*exp(1j*0.5), ...        % 强主径 (带随机相位)
               0.3*exp(-1j*1.2), ...       % 极近弱径 (带相位)
               0.6*exp(1j*2.1)];           % 远端中等径

N_rx = 1000;         % 接收信号总长度
y_rx = zeros(N_rx, 1); % 初始化纯净接收信号

% 叠加多径生成接收信号
for k = 1:length(true_delays)
    idx_start = true_delays(k);
    idx_end   = idx_start + N_seq - 1;
    y_rx(idx_start:idx_end) = y_rx(idx_start:idx_end) + true_amps(k) * s_tx;
end

% 加入高斯白噪声 (SNR = 15dB)
noise = (randn(N_rx,1) + 1j*randn(N_rx,1)) / sqrt(2);
y_rx = y_rx + 0.1 * noise; 

fprintf('仿真接收信号生成完毕。信号长度: %d, 发射模板长度: %d\n', N_rx, N_seq);

%% 2. SAGE 算法系统参数设置
L_paths = 3;         % 假设我们需要提取 3 条径 (实际中可以设多一点)
max_iter = 15;       % SAGE E-M 迭代次数
energy_s = s_tx' * s_tx; % 发射模板的能量 (归一化后理论上为1)

% 初始化 SAGE 参数存储矩阵
est_delays = zeros(L_paths, 1); % 存储整数时延
est_amps   = zeros(L_paths, 1); % 存储复数幅度 (极其重要！)

%% 3. Step 1: 串行干扰消除 (SIC) 粗初始化
fprintf('\n--- 开始 SIC 粗初始化 ---\n');
residual = y_rx; % 初始残差就是原始接收信号

for l = 1:L_paths
    % 在复数域计算残差与发射波形的滑动互相关 (内积)
    % 搜索范围：从第 1 个点到 N_rx - N_seq + 1 个点
    search_len = N_rx - N_seq + 1;
    corr_vals = zeros(search_len, 1);
    
    for tau = 1:search_len
        % 截取当前残差信号的一个窗口
        windowed_res = residual(tau : tau + N_seq - 1);
        % 计算复数内积 (注意 s_tx 需要共轭转置 ')
        corr_vals(tau) = s_tx' * windowed_res; 
    end
    
    % 寻找复数互相关绝对值最大的位置
    [~, best_tau] = max(abs(corr_vals));
    
    % 提取复数幅度！这是抹除旁瓣的关键！
    best_amp = corr_vals(best_tau) / energy_s; 
    
    est_delays(l) = best_tau;
    est_amps(l)   = best_amp;
    
    % --- 核心魔术：相干重构与相减 (去除该径的所有旁瓣) ---
    s_reconstructed = zeros(N_rx, 1);
    s_reconstructed(best_tau : best_tau + N_seq - 1) = best_amp * s_tx;
    residual = residual - s_reconstructed; % 复数相减，旁瓣灰飞烟灭！
    
    fprintf('初始化路径 %d: 时延 = %d, 幅度 = %.3f, 相位 = %.2f rad\n', ...
        l, best_tau, abs(best_amp), angle(best_amp));
end

%% 4. Step 2: SAGE 核心 EM 迭代 (精细逼近)
fprintf('\n--- 开始 SAGE EM 迭代 ---\n');
for iter = 1:max_iter
    for l = 1:L_paths
        % 1. E-step: 构造只有第 l 条径的“纯净观测信号” x_l
        % 先把之前减掉的第 l 条径的波形加回来
        old_tau = est_delays(l);
        old_amp = est_amps(l);
        s_old = zeros(N_rx, 1);
        s_old(old_tau : old_tau + N_seq - 1) = old_amp * s_tx;
        
        x_l = residual + s_old; % 此时 x_l 里包含了第 l 条径 + 底噪
        
        % 2. M-step: 在旧时延附近进行局部精细搜索
        search_radius = 5; % 局部搜索半径 (因为初始化已经很准了)
        search_range = max(1, old_tau - search_radius) : min(N_rx - N_seq + 1, old_tau + search_radius);
        
        best_p_val = -1;
        best_tau_new = old_tau;
        best_amp_new = old_amp;
        
        for tau_test = search_range
            windowed_x = x_l(tau_test : tau_test + N_seq - 1);
            proj = s_tx' * windowed_x; % 依然是复数内积
            
            if abs(proj) > best_p_val
                best_p_val = abs(proj);
                best_tau_new = tau_test;
                best_amp_new = proj / energy_s;
            end
        end
        
        % 更新第 l 条径的参数
        est_delays(l) = best_tau_new;
        est_amps(l)   = best_amp_new;
        
        % 3. 更新全局残差 (为下一条径做准备)
        s_new = zeros(N_rx, 1);
        s_new(best_tau_new : best_tau_new + N_seq - 1) = best_amp_new * s_tx;
        residual = x_l - s_new;
    end
end
fprintf('SAGE 迭代完成！\n');

%% 5. 结果可视化与学术作图
% 为了对比，计算传统互相关的“信道冲激响应包络” (就像你之前的 h)
h_traditional = zeros(N_rx - N_seq + 1, 1);
for tau = 1:length(h_traditional)
    h_traditional(tau) = abs(s_tx' * y_rx(tau : tau + N_seq - 1));
end
h_traditional = h_traditional / energy_s;

% 构建 SAGE 提取出的稀疏信道
h_sage_sparse = zeros(length(h_traditional), 1);
for l = 1:L_paths
    h_sage_sparse(est_delays(l)) = abs(est_amps(l));
end

figure('Position', [100, 100, 1000, 500]);
% 绘制传统互相关包络 (会看到很高的旁瓣)
plot(1:length(h_traditional), h_traditional, 'Color', [0.7 0.7 0.7], 'LineWidth', 1.5); hold on;

% 绘制真实的信道冲激响应 (红色的针)
stem(true_delays, abs(true_amps), 'r', 'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 8);

% 绘制 SAGE 提取的信道冲激响应 (蓝色的针，带三角)
stem(find(h_sage_sparse>0), h_sage_sparse(h_sage_sparse>0), 'b', 'LineWidth', 2, 'Marker', '^', 'MarkerSize', 8);

title('基于复基带信号的 SAGE 多径提取 vs. 传统互相关', 'FontSize', 14);
xlabel('时延 (采样点索引)', 'FontSize', 12);
ylabel('幅度', 'FontSize', 12);
legend('传统滑动互相关包络 (可见严重旁瓣污染)', '真实信道多径', 'SAGE 提取结果 (完美避开旁瓣)', 'Location', 'northeast', 'FontSize', 11);
grid on;
xlim([20, 250]);