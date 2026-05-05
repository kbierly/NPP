% experiment_full.m
warning('off', 'manopt:getHessian:approx');
rng(42);

%% Parameters
sizes      = [10, 20];
n_ratios   = [1, 0.5];
num_trials = 3;
num_starts = 4;

trial_colors = [0.2 0.6 1.0;
                1.0 0.5 0.1;
                0.2 0.8 0.2];

%% Part 1: Convergence trajectories + table data
conv_data     = cell(length(sizes), length(n_ratios), num_trials);
res_mean      = zeros(length(sizes), length(n_ratios));
time_mean     = zeros(length(sizes), length(n_ratios));
norm_mean     = zeros(length(sizes), length(n_ratios));
gradnorm_mean = zeros(length(sizes), length(n_ratios));

for si = 1:length(sizes)
    m = sizes(si);
    for ri = 1:length(n_ratios)
        n = max(1, floor(n_ratios(ri) * m));
        fprintf('Part 1: m=%d, n=%d...\n', m, n);
        res_vals      = zeros(num_trials, 1);
        time_vals     = zeros(num_trials, 1);
        norm_vals     = zeros(num_trials, 1);
        gradnorm_vals = zeros(num_trials, 1);
        for t = 1:num_trials
            X = randn(m,n) + 1i*randn(m,n);
            Y = randn(m,n) + 1i*randn(m,n);
            tic;
            [A, ~, ~, hist] = normal_procrustes_history(X, Y);
            time_vals(t)     = toc;
            res_vals(t)      = hist.residual(end);
            norm_vals(t)     = norm(A*A' - A'*A, 'fro');
            gradnorm_vals(t) = hist.gradnorm(end);
            conv_data{si, ri, t} = hist;
        end
        res_mean(si,ri)      = mean(res_vals);
        time_mean(si,ri)     = mean(time_vals);
        norm_mean(si,ri)     = mean(norm_vals);
        gradnorm_mean(si,ri) = mean(gradnorm_vals);
    end
end

%% Part 2: Known global minimum test
fprintf('Part 2: Known global minimum tests...\n');
known_res  = zeros(length(sizes), length(n_ratios), num_starts);
known_conv = cell(length(sizes), length(n_ratios), num_starts);

for si = 1:length(sizes)
    m = sizes(si);
    for ri = 1:length(n_ratios)
        n = max(1, floor(n_ratios(ri) * m));
        [U_true, ~] = qr(randn(m,m) + 1i*randn(m,m));
        d_true  = randn(m,1) + 1i*randn(m,1);
        A_true  = U_true * diag(d_true) * U_true';
        X = randn(m,n) + 1i*randn(m,n);
        Y = A_true * X;
        fprintf('  m=%d, n=%d:\n', m, n);
        for s = 1:num_starts
            [~, ~, ~, hist] = normal_procrustes_history(X, Y);
            known_res(si,ri,s)  = hist.residual(end);
            known_conv{si,ri,s} = hist;
            fprintf('    start %d: residual=%.2e\n', s, known_res(si,ri,s));
        end
    end
end

%% Part 3: Ruhe comparison (X = I, 1 trial per size)
% Uses specialized NPP solver for X=I: since U is unitary, ||(U*X)_i||^2 = 1
% for all i, so Phi = I identically. This simplifies gradient and Hessian.
fprintf('Part 3: Ruhe comparison...\n');
ruhe_sizes  = [5, 10, 15, 20];
num_ruhe    = 1;
ruhe_res    = zeros(length(ruhe_sizes), num_ruhe);
ruhe_time   = zeros(length(ruhe_sizes), num_ruhe);
npp_res     = zeros(length(ruhe_sizes), num_ruhe);
npp_time    = zeros(length(ruhe_sizes), num_ruhe);
ruhe_conv   = cell(length(ruhe_sizes), num_ruhe);
npp_conv_r3 = cell(length(ruhe_sizes), num_ruhe);

for si = 1:length(ruhe_sizes)
    m = ruhe_sizes(si);
    fprintf('  m=%d...\n', m);
    for t = 1:num_ruhe
        Y = randn(m,m) + 1i*randn(m,m);

        % NPP with X=I: use specialized solver with simplified grad/Hess
        tic;
        [~, ~, ~, hist_npp] = normal_procrustes_history_identity(Y);
        npp_time(si,t) = toc;
        npp_res(si,t)  = hist_npp.residual(end);
        npp_conv_r3{si,t} = hist_npp;

        % Ruhe's algorithm
        tic;
        [~, hist_ruhe] = ruhe_normal(Y);
        ruhe_time(si,t) = toc;
        ruhe_res(si,t)  = hist_ruhe.residual(end);
        ruhe_conv{si,t} = hist_ruhe;

        fprintf('    NPP:  res=%.4f, time=%.3fs\n', npp_res(si,t),  npp_time(si,t));
        fprintf('    Ruhe: res=%.4f, time=%.3fs\n', ruhe_res(si,t), ruhe_time(si,t));
    end
end

%% Figure 1: Convergence + gradient norm
ncols = length(n_ratios);
nrows = length(sizes) * 2;
figure('Name','Figure 1','Position',[100 100 900 950]);

all_grad_vals = [];
for si = 1:length(sizes)
    for ri = 1:length(n_ratios)
        for t = 1:num_trials
            h = conv_data{si,ri,t};
            all_grad_vals = [all_grad_vals; h.gradnorm(h.gradnorm > 0)];
        end
    end
end
grad_ylim = [min(all_grad_vals)*0.5, max(all_grad_vals)*2];

for si = 1:length(sizes)
    m = sizes(si);
    for ri = 1:length(n_ratios)
        n = max(1, floor(n_ratios(ri) * m));
        sub_res_vals = [];
        for t = 1:num_trials
            sub_res_vals = [sub_res_vals; conv_data{si,ri,t}.residual];
        end
        res_ylim_sub = [min(sub_res_vals)*0.95, max(sub_res_vals)*1.05];

        subplot(nrows, ncols, (si-1)*ncols + ri);
        for t = 1:num_trials
            h = conv_data{si, ri, t};
            plot(h.time, h.residual, '-', 'Color', trial_colors(t,:), 'LineWidth', 1.5);
            hold on;
        end
        ylim(res_ylim_sub);
        ylabel('||AX-Y||^2_F');
        title(sprintf('m=%d, n=%d', m, n));
        grid on;
        if si == 1 && ri == 1
            legend(arrayfun(@(t) sprintf('Trial %d', t), 1:num_trials, ...
                'UniformOutput', false), 'Location', 'northeast', 'FontSize', 8);
        end

        subplot(nrows, ncols, (length(sizes)+si-1)*ncols + ri);
        for t = 1:num_trials
            h = conv_data{si, ri, t};
            semilogy(h.time, max(h.gradnorm, 1e-16), '-', 'Color', trial_colors(t,:), 'LineWidth', 1.5);
            hold on;
        end
        ylim(grad_ylim);
        ylabel('grad norm');
        xlabel('CPU time (s)');
        title(sprintf('Grad norm, m=%d, n=%d', m, n));
        grid on;
    end
end

annotation('line', [0.05 0.97], [0.495 0.495], 'Color', [0.7 0.7 0.7], 'LineWidth', 1.5);
annotation('textbox', [0.01 0.965 0.98 0.025], 'String', 'TOP: Residual ||AX-Y||^2_F  (linear)', ...
    'HorizontalAlignment', 'center', 'EdgeColor', 'none', 'FontWeight', 'bold', 'FontSize', 9, 'FitBoxToText', 'off');
annotation('textbox', [0.01 0.01 0.98 0.025], 'String', 'BOTTOM: Riemannian Gradient Norm  (log)', ...
    'HorizontalAlignment', 'center', 'EdgeColor', 'none', 'FontWeight', 'bold', 'FontSize', 9, 'FitBoxToText', 'off');
sgtitle('Figure 1: Convergence -- Gaussian inputs, X and Y random per trial');
saveas(gcf, 'fig1_convergence.png');
fprintf('Saved fig1_convergence.png\n');

%% Figure 2: Known global minimum recovery -- convergence trajectories only
% True residual = 0, so curves converging to ~0 confirm global min found.
% Curves plateauing above 0 confirm local minima.
start_colors2 = lines(num_starts);
num_configs   = length(sizes) * length(n_ratios);
figure('Name','Figure 2','Position',[100 100 1000 400]);
plot_idx = 1;
for si = 1:length(sizes)
    m = sizes(si);
    for ri = 1:length(n_ratios)
        n = max(1, floor(n_ratios(ri) * m));
        subplot(1, num_configs, plot_idx);
        for s = 1:num_starts
            h = known_conv{si,ri,s};
            semilogy(h.time, max(h.residual, 1e-16), '-', ...
                'Color', start_colors2(s,:), 'LineWidth', 1.5);
            hold on;
        end
        xlabel('CPU time (s)');
        ylabel('||AX-Y||^2_F');
        title(sprintf('m=%d, n=%d', m, n));
        grid on;
        if plot_idx == 1
            legend(arrayfun(@(s) sprintf('Start %d', s), 1:num_starts, ...
                'UniformOutput', false), 'Location', 'northeast', 'FontSize', 8);
        end
        plot_idx = plot_idx + 1;
    end
end
sgtitle({'Figure 2: Known global minimum recovery (true min = 0)'});
saveas(gcf, 'fig2_known_min.png');
fprintf('Saved fig2_known_min.png\n');

%% Figure 3: NPP vs Ruhe comparison
npp_color  = [0.2 0.5 0.9];
ruhe_color = [0.9 0.4 0.2];
figure('Name','Figure 3','Position',[100 100 1000 400]);
for si = 1:length(ruhe_sizes)
    m = ruhe_sizes(si);
    subplot(1, length(ruhe_sizes), si);
    for t = 1:num_ruhe
        h_npp  = npp_conv_r3{si,t};
        h_ruhe = ruhe_conv{si,t};
        plot(h_npp.time,  h_npp.residual,  '-', 'Color', npp_color,  'LineWidth', 1.5);
        hold on;
        plot(h_ruhe.time, h_ruhe.residual, '-', 'Color', ruhe_color, 'LineWidth', 1.5);
    end
    xlabel('CPU time (s)');
    ylabel('||A-Y||^2_F');
    title(sprintf('m=%d', m));
    grid on;
    if si == 1
        legend('NPP (ours)', 'Ruhe', 'Location', 'northeast', 'FontSize', 8);
    end
end
sgtitle('Figure 3: NPP vs Ruhe (X = I) -- Gaussian inputs');
saveas(gcf, 'fig3_ruhe.png');
fprintf('Saved fig3_ruhe.png\n');

%% Summary table
fprintf('\n========== Summary Table ==========\n');
fprintf('Complex Gaussian inputs. Results averaged over %d trials.\n\n', num_trials);
fprintf('  Residual  = mean ||AX-Y||_F^2 at convergence (from history)\n');
fprintf('  Time      = mean total CPU time (seconds)\n');
fprintf('  Grad norm = mean Riemannian gradient norm at convergence\n');
fprintf('  Norm err  = mean ||AA*-A*A||_F (should be ~0)\n\n');
fprintf('%-6s %-6s %-14s %-12s %-12s %-12s\n', ...
    'm', 'n', 'Residual', 'Time(s)', 'Grad norm', 'Norm err');
fprintf('%s\n', repmat('-', 1, 64));
for si = 1:length(sizes)
    m = sizes(si);
    for ri = 1:length(n_ratios)
        n = max(1, floor(n_ratios(ri) * m));
        fprintf('%-6d %-6d %-14.4f %-12.3f %-12.2e %-12.2e\n', ...
            m, n, res_mean(si,ri), time_mean(si,ri), ...
            gradnorm_mean(si,ri), norm_mean(si,ri));
    end
end
fprintf('\nAll figures saved as PNG in current directory.\n');

%% ========== NPP Specialized Solver for X = I ==========
function [A, U, fval, history] = normal_procrustes_history_identity(Y)
% Solves min_{A normal} ||A - Y||_F^2 (i.e. NPP with X = I).
% Exploits that Phi = I identically when X = I (unitary rows have norm 1),
% so all Phi^{-1} terms drop out and delta_Phi = 0.
%
% Simplified gradient (X=I, Phi=I):
%   grad_U f = 2(Y*U*Lambda + Y*U*Lambda* - U*|Lambda|^2)
% where Lambda = diag(U*YU) (diagonal of U*YU).
%
% Simplified Hessian (X=I, Phi=I, delta_Phi=0):
%   (1/2) Hess_U f[W] = Y*W*Lambda + Y*U*delta_Lambda
%                     + Y'*W*Lambda* + Y'*U*delta_Lambda*
%                     - W*|Lambda|^2 - U*delta(|Lambda|^2)
% where delta_Lambda = diag(W*YU + U*YW).

m = size(Y, 1);

time_list     = [];
cost_list     = [];
gradnorm_list = [];
t0 = tic;

M = unitaryfactory(m);

problem.M = M;
problem.cost  = @(Uvar)       -f_identity(Uvar, Y);
problem.egrad = @(Uvar)       -euclidean_grad_identity(Uvar, Y);
problem.ehess = @(Uvar, Wvar) -euclidean_hess_identity(Uvar, Wvar, Y, M);

options.statsfun  = @(problem, x, stats, store) record_stats(stats);
options.verbosity = 0;
options.Delta_bar = 1000;
options.maxiter   = 2000;
warning('off', 'manopt:getHessian:approx');

x0 = M.rand();
[U, neg_fval, ~, ~] = trustregions(problem, x0, options);
fval = -neg_fval;

if isstruct(U), U = U.U; end

history.time     = time_list - time_list(1);
% Residual = ||Y||_F^2 - f(U) by Proposition 2 with X=I
history.residual = norm(Y,'fro')^2 + cost_list;
history.gradnorm = gradnorm_list;

% Recover A: Lambda = diag(U*YU), D = diag(Lambda), A = UDU*
Lambda = diag(U'*Y*U);
A = U * diag(Lambda) * U';

    function stats = record_stats(stats)
        time_list(end+1, 1)     = toc(t0);
        cost_list(end+1, 1)     = stats.cost;
        gradnorm_list(end+1, 1) = stats.gradnorm;
    end
end

function val = f_identity(U, Y)
% f(U) = sum_i |(U*Y)_i * U_i^*|^2 = sum_i |(U*YU)_ii|^2
% i.e. squared Frobenius norm of diagonal of U*YU
    if isstruct(U), U = U.U; end
    Lambda = diag(U'*Y*U); % diagonal entries of U*YU
    val = sum(abs(Lambda).^2);
end

function G = euclidean_grad_identity(U, Y)
% Euclidean gradient when X=I, Phi=I:
%   grad = 2(Y*' U Lambda + Y U Lambda* - U |Lambda|^2)
% where Lambda = diag(U*YU)
    if isstruct(U), U = U.U; end
    Lambda     = diag(U'*Y*U);       % m x 1 vector of diagonal entries
    Lam        = diag(Lambda);        % diagonal matrix
    LamC       = diag(conj(Lambda));  % Lambda*
    LamAbs2    = diag(abs(Lambda).^2);% |Lambda|^2
    G = 2*(Y'*U*Lam + Y*U*LamC - U*LamAbs2);
end

function H = euclidean_hess_identity(U, W, Y, M)
% Euclidean Hessian-vector product when X=I, Phi=I, delta_Phi=0:
%   (1/2) Hess[W] = Y'*W*Lambda + Y'*U*dLambda
%                 + Y*W*Lambda* + Y*U*dLambda*
%                 - W*|Lambda|^2 - U*(dLambda*Lambda* + Lambda*dLambda*)
% where dLambda = diag(W*YU + U*YW)
    if isstruct(U), U = U.U; end
    W = M.tangent2ambient(U, W);

    Lambda  = diag(U'*Y*U);              % diagonal of U*YU
    dLambda = diag(W'*Y*U + U'*Y*W);    % delta_Lambda = diag(W*YU + U*YW)

    Lam      = diag(Lambda);
    LamC     = diag(conj(Lambda));
    dLam     = diag(dLambda);
    dLamC    = diag(conj(dLambda));
    LamAbs2  = diag(abs(Lambda).^2);
    dLamAbs2 = dLam*diag(conj(Lambda)) + diag(Lambda)*dLamC; % delta(|Lambda|^2)

    H = 2*(Y'*W*Lam  + Y'*U*dLam  + ...
           Y*W*LamC  + Y*U*dLamC  - ...
           W*LamAbs2 - U*dLamAbs2);
end

%% ========== Ruhe's Algorithm ==========
function [N, history] = ruhe_normal(Y)
% Ruhe's Algorithm J (Ruhe 1987): closest normal matrix.
% One rotation per step, pivot = max |h_jk|.
%
% Input:  Y -- m x m complex matrix
% Output: N -- closest normal matrix
%         history -- .time and .residual per step

m = size(Y, 1);
A = Y;         % working copy, transformed by rotations A <- R^H A R
Q = eye(m);    % accumulated unitary: Y = Q * A * Q^H at all times
max_sweeps = 5000;
tol = 1e-10;

time_list = [];
res_list  = [];
t0 = tic;

for sweep = 1:max_sweeps

    % ----------------------------------------------------------------
    % STEP 1 (Algorithm J): find pivot (j,k) with maximum |h_jk|
    % h_jk = e^{-i*theta} a_jk + e^{i*theta} conj(a_kj)  (Ruhe step 3)
    % theta = 0.5 * arg(det(B)) where B is the centered 2x2 block (Ruhe step 2)
    % Save theta and h_jk (complex) for the winner to avoid recomputation
    % ----------------------------------------------------------------
    best_hjk_abs = 0;
    best_j   = 1; best_k = 2;
    best_hjk = 0; % complex value of h_jk for winning pair
    best_th  = 0; % theta for winning pair

    for j = 1:m
        for k = j+1:m
            ajj = A(j,j); akk = A(k,k);
            ajk = A(j,k); akj = A(k,j);

            % Step 2: shift and phase
            mval = (ajj + akk) / 2;
            B    = [ajj-mval, ajk; akj, akk-mval];
            th   = 0.5 * angle(det(B));

            % Step 3: tangential component h_jk
            hjk  = exp(-1i*th)*ajk + exp(1i*th)*conj(akj);

            if abs(hjk) > best_hjk_abs
                best_hjk_abs = abs(hjk);
                best_hjk     = hjk;
                best_th      = th;
                best_j = j; best_k = k;
            end
        end
    end

    % Convergence: h_jk -> 0 iff A is a Delta-H matrix (Ruhe Section 3)
    if best_hjk_abs < tol
        break;
    end

    j = best_j; k = best_k;
    ajj = A(j,j); akk = A(k,k);

    % ----------------------------------------------------------------
    % STEP 3: compute rotation angle phi and phase alpha
    % Using saved theta and h_jk from pivot search
    % phi = 0.5 * arctan(|h_jk| / Re(e^{-i*theta}(a_jj - a_kk)))
    % alpha = arg(h_jk)
    % ----------------------------------------------------------------
    denom = real(exp(-1i*best_th)*(ajj - akk));
    phi   = 0.5 * atan2(best_hjk_abs, denom);
    alpha = angle(best_hjk);

    % ----------------------------------------------------------------
    % STEP 4: build rotation matrix R and apply A <- R^H A R
    % R has nontrivial 2x2 block at (j,k):
    %   [cos(phi),             -e^{i*alpha} sin(phi)]
    %   [e^{-i*alpha} sin(phi),  cos(phi)           ]
    % ----------------------------------------------------------------
    R = eye(m);
    R(j,j) =  cos(phi);
    R(j,k) = -exp(1i*alpha) * sin(phi);
    R(k,j) =  exp(-1i*alpha) * sin(phi);
    R(k,k) =  cos(phi);
    A = R' * A * R;
    Q = Q * R; % update accumulated rotation: Y = Q * A * Q^H

    % ----------------------------------------------------------------
    % Record residual: by unitary invariance ||N - Y||_F^2 = ||A - diag(A)||_F^2
    % since Y = Q*A*Q^H and N = Q*diag(A)*Q^H
    % ----------------------------------------------------------------
    res = norm(A - diag(diag(A)), 'fro')^2;
    time_list(end+1, 1) = toc(t0); %#ok<AGROW>
    res_list(end+1, 1)  = res;     %#ok<AGROW>
end

% Recover N = Q * diag(A_final) * Q^H
N = Q * diag(diag(A)) * Q';

if isempty(time_list)
    history.time     = 0;
    history.residual = norm(A - diag(diag(A)), 'fro')^2;
else
    history.time     = time_list - time_list(1);
    history.residual = res_list;
end
end