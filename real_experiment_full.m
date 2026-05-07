% real_experiment_full.m
% Numerical experiments for the Real Normal Procrustes Problem.
% Compares our Riemannian optimization approach against Guglielmi-Scalone (2019).
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

%% =========================================================================
%% Part 1: Convergence trajectories + table data (real Gaussian inputs)
%% =========================================================================
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
            X = randn(m, n);
            Y = randn(m, n);
            tic;
            [A, ~, ~, hist] = real_normal_procrustes_history(X, Y);
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

%% =========================================================================
%% Part 2: Known global minimum recovery
%% All four sizes for convergence plots.
%% Detailed eigenvalue printout for 5x5 and 10x5 separately.
%% =========================================================================
fprintf('\nPart 2: Known global minimum tests...\n');
known_conv = cell(length(sizes), length(n_ratios), num_starts);

for si = 1:length(sizes)
    m = sizes(si);
    for ri = 1:length(n_ratios)
        n = max(1, floor(n_ratios(ri) * m));
        fprintf('  m=%d, n=%d:\n', m, n);
        A_true    = make_real_normal(m);
        X         = randn(m, n);
        Y         = A_true * X;
        for s = 1:num_starts
            [A_rec, ~, ~, hist] = real_normal_procrustes_history(X, Y);
            known_conv{si,ri,s} = hist;
            proj_err = norm((A_rec - A_true)*X,'fro');
            fprintf('    start %d: residual=%.2e,  ||(A_rec-A_true)X||=%.2e', ...
                s, hist.residual(end), proj_err);
            if proj_err < 1e-5, fprintf('  (MATCH)\n'); else, fprintf('  (differ)\n'); end
        end
        fprintf('\n');
    end
end

% Detailed printout for 5x5 and 10x5
fprintf('--- Detailed printout: 5x5 ---\n');
print_known_min_detail(5, 5);
fprintf('\n--- Detailed printout: 10x5 ---\n');
print_known_min_detail(10, 5);

%% =========================================================================
%% Part 3: Comparison with Guglielmi-Scalone (X = I, real case)
%% =========================================================================
fprintf('\nPart 3: Guglielmi-Scalone comparison (X=I, real Gaussian)...\n');
gs_sizes = [5, 10, 15, 20];
num_runs = 1;

gs_res   = zeros(length(gs_sizes), num_runs);
gs_time  = zeros(length(gs_sizes), num_runs);
npp_res  = zeros(length(gs_sizes), num_runs);
npp_time = zeros(length(gs_sizes), num_runs);
gs_conv  = cell(length(gs_sizes), num_runs);
npp_conv = cell(length(gs_sizes), num_runs);

for si = 1:length(gs_sizes)
    m = gs_sizes(si);
    fprintf('  m=%d...\n', m);
    for t = 1:num_runs
        Y_real = randn(m, m);

        % Our algorithm (X=I, real) -- uses specialized solver exploiting X=I
        tic;
        [A_npp, ~, ~, hist_npp] = real_npp_identity(Y_real);
        npp_time(si,t) = toc;
        npp_res(si,t)  = hist_npp.residual(end);
        npp_conv{si,t} = hist_npp;

        % Guglielmi-Scalone
        tic;
        [A_gs, hist_gs] = guglielmi_scalone(Y_real);
        gs_time(si,t) = toc;
        gs_res(si,t)  = norm(A_gs - Y_real,'fro')^2;
        gs_conv{si,t} = hist_gs;

        fprintf('    NPP: res=%.4f, time=%.3fs,  norm_err=%.2e\n', ...
            npp_res(si,t), npp_time(si,t), norm(A_npp*A_npp'-A_npp'*A_npp,'fro'));
        fprintf('    GS:  res=%.4f, time=%.3fs,  norm_err=%.2e\n', ...
            gs_res(si,t), gs_time(si,t), norm(A_gs*A_gs'-A_gs'*A_gs,'fro'));

        if m == 5
            fprintf('\n  --- 5x5 detailed comparison ---\n');
            fprintf('  NPP eigenvalues:   '); print_eigs(eig(A_npp));
            fprintf('  GS  eigenvalues:   '); print_eigs(eig(A_gs));
            fprintf('  NPP normality err: %.2e\n', norm(A_npp*A_npp'-A_npp'*A_npp,'fro'));
            fprintf('  GS  normality err: %.2e\n', norm(A_gs*A_gs' -A_gs' *A_gs, 'fro'));
            fprintf('  NPP ||A-Y||_F^2:   %.4f\n', norm(A_npp-Y_real,'fro')^2);
            fprintf('  GS  ||A-Y||_F^2:   %.4f\n', norm(A_gs -Y_real,'fro')^2);
        end
    end
end

%% =========================================================================
%% Figure 1: Convergence + gradient norm
%% =========================================================================
ncols = length(n_ratios);
nrows = length(sizes) * 2;
figure('Name','Figure 1 Real','Position',[100 100 900 950]);

all_grad_vals = [];
for si = 1:length(sizes)
    for ri = 1:length(n_ratios)
        for t = 1:num_trials
            h = conv_data{si,ri,t};
            all_grad_vals = [all_grad_vals; h.gradnorm(h.gradnorm > 0)]; %#ok<AGROW>
        end
    end
end
grad_ylim = [min(all_grad_vals)*0.5, max(all_grad_vals)*2];

for si = 1:length(sizes)
    m = sizes(si);
    for ri = 1:length(n_ratios)
        n = max(1, floor(n_ratios(ri)*m));
        sub_res_vals = [];
        for t = 1:num_trials
            sub_res_vals = [sub_res_vals; conv_data{si,ri,t}.residual]; %#ok<AGROW>
        end
        res_ylim_sub = [min(sub_res_vals)*0.95, max(sub_res_vals)*1.05];

        subplot(nrows, ncols, (si-1)*ncols + ri);
        for t = 1:num_trials
            h = conv_data{si,ri,t};
            plot(h.time, h.residual, '-', 'Color', trial_colors(t,:), 'LineWidth', 1.5);
            hold on;
        end
        ylim(res_ylim_sub); ylabel('||AX-Y||^2_F'); title(sprintf('m=%d, n=%d',m,n)); grid on;
        if si==1 && ri==1
            legend(arrayfun(@(t) sprintf('Trial %d',t), 1:num_trials,'UniformOutput',false),...
                'Location','northeast','FontSize',8);
        end

        subplot(nrows, ncols, (length(sizes)+si-1)*ncols + ri);
        for t = 1:num_trials
            h = conv_data{si,ri,t};
            semilogy(h.time, max(h.gradnorm,1e-16), '-', 'Color', trial_colors(t,:), 'LineWidth', 1.5);
            hold on;
        end
        ylim(grad_ylim); ylabel('grad norm'); xlabel('CPU time (s)');
        title(sprintf('Grad norm, m=%d, n=%d',m,n)); grid on;
    end
end
annotation('line',[0.05 0.97],[0.495 0.495],'Color',[0.7 0.7 0.7],'LineWidth',1.5);
annotation('textbox',[0.01 0.965 0.98 0.025],'String','TOP: Residual ||AX-Y||^2_F  (linear)',...
    'HorizontalAlignment','center','EdgeColor','none','FontWeight','bold','FontSize',9,'FitBoxToText','off');
annotation('textbox',[0.01 0.01 0.98 0.025],'String','BOTTOM: Riemannian Gradient Norm  (log)',...
    'HorizontalAlignment','center','EdgeColor','none','FontWeight','bold','FontSize',9,'FitBoxToText','off');
sgtitle('Figure 1 (Real): Convergence -- real Gaussian inputs');
saveas(gcf,'fig1_real_convergence.png');
fprintf('Saved fig1_real_convergence.png\n');

%% =========================================================================
%% Figure 2: Known global minimum recovery
%% =========================================================================
start_colors2 = lines(num_starts);
num_configs   = length(sizes)*length(n_ratios);
figure('Name','Figure 2 Real','Position',[100 100 1000 400]);
plot_idx = 1;
for si = 1:length(sizes)
    m = sizes(si);
    for ri = 1:length(n_ratios)
        n = max(1, floor(n_ratios(ri)*m));
        subplot(1, num_configs, plot_idx);
        for s = 1:num_starts
            h = known_conv{si,ri,s};
            semilogy(h.time, max(h.residual,1e-16), '-', 'Color', start_colors2(s,:), 'LineWidth', 1.5);
            hold on;
        end
        xlabel('CPU time (s)'); ylabel('||AX-Y||^2_F'); title(sprintf('m=%d, n=%d',m,n)); grid on;
        if plot_idx==1
            legend(arrayfun(@(s) sprintf('Start %d',s),1:num_starts,'UniformOutput',false),...
                'Location','northeast','FontSize',8);
        end
        plot_idx = plot_idx + 1;
    end
end
sgtitle('Figure 2 (Real): Known global minimum recovery (true min = 0)');
saveas(gcf,'fig2_real_known_min.png');
fprintf('Saved fig2_real_known_min.png\n');

%% =========================================================================
%% Figure 3: NPP vs Guglielmi-Scalone
%% =========================================================================
npp_color = [0.2 0.5 0.9];
gs_color  = [0.8 0.2 0.6];
figure('Name','Figure 3 Real','Position',[100 100 1000 400]);
for si = 1:length(gs_sizes)
    m = gs_sizes(si);
    subplot(1, length(gs_sizes), si);
    plot(npp_conv{si,1}.time, npp_conv{si,1}.residual, '-', 'Color', npp_color, 'LineWidth', 1.5);
    hold on;
    plot(gs_conv{si,1}.time,  gs_conv{si,1}.residual,  '-', 'Color', gs_color,  'LineWidth', 1.5);
    xlabel('CPU time (s)'); ylabel('||A-Y||^2_F'); title(sprintf('m=%d',m)); grid on;
    if si==1, legend('NPP (ours)','Guglielmi-Scalone','Location','northeast','FontSize',8); end
end
sgtitle('Figure 3 (Real): NPP vs Guglielmi-Scalone (X=I)');
saveas(gcf,'fig3_real_gs.png');
fprintf('Saved fig3_real_gs.png\n');

%% Summary table
fprintf('\n========== Summary Table (Real NPP) ==========\n');
fprintf('Real Gaussian inputs. Results averaged over %d trials.\n\n', num_trials);
fprintf('%-6s %-6s %-14s %-12s %-12s %-12s\n','m','n','Residual','Time(s)','Grad norm','Norm err');
fprintf('%s\n', repmat('-',1,64));
for si = 1:length(sizes)
    m = sizes(si);
    for ri = 1:length(n_ratios)
        n = max(1, floor(n_ratios(ri)*m));
        fprintf('%-6d %-6d %-14.4f %-12.3f %-12.2e %-12.2e\n', ...
            m, n, res_mean(si,ri), time_mean(si,ri), gradnorm_mean(si,ri), norm_mean(si,ri));
    end
end
fprintf('\nAll figures saved.\n');


%% =========================================================================
%% Helper functions
%% =========================================================================

function A = make_real_normal(m)
    Delta = zeros(m,m);
    for i = 1:floor(m/2)
        a = randn; b = randn;
        Delta(2*i-1:2*i,2*i-1:2*i) = [a,b;-b,a];
    end
    if mod(m,2)==1, Delta(m,m) = randn; end
    [Q,~] = qr(randn(m,m));
    A = Q * Delta * Q';
end

function ev = sort_eigs(ev)
    [~,idx] = sortrows([real(ev), imag(ev)]);
    ev = ev(idx);
end

function print_eigs(ev)
    ev = sort_eigs(ev);
    for k = 1:length(ev)
        fprintf('%.4f%+.4fi  ', real(ev(k)), imag(ev(k)));
    end
    fprintf('\n');
end

function print_known_min_detail(m, n)
    A_true    = make_real_normal(m);
    X         = randn(m,n);
    Y         = A_true * X;
    best_res  = inf;
    A_best    = [];
    for s = 1:4
        [A_rec, ~, ~, hist] = real_normal_procrustes_history(X, Y);
        if hist.residual(end) < best_res
            best_res = hist.residual(end);
            A_best   = A_rec;
        end
        % For n<m, eigenvalue recovery is non-identifiable (infinitely many
        % normal A satisfy AX=Y). Report residual and projection agreement instead.
        proj_err = norm((A_rec - A_true)*X,'fro');
        fprintf('  start %d: residual=%.2e,  ||(A_rec-A_true)X||_F=%.2e\n', ...
            s, hist.residual(end), proj_err);
    end
    fprintf('  A_true eigenvalues: '); print_eigs(eig(A_true));
    fprintf('  A_best eigenvalues: '); print_eigs(eig(A_best));
    fprintf('  ||(A_best-A_true)X||_F = %.2e  (should be ~0)\n', norm((A_best-A_true)*X,'fro'));
    fprintf('  ||A_best - A_true||_F  = %.2e  (non-identifiable for n<m)\n', norm(A_best-A_true,'fro'));
    fprintf('  Normality err of A_best: %.2e\n', norm(A_best*A_best'-A_best'*A_best,'fro'));
end


%% =========================================================================
%% Specialized Real NPP solver for X = I
%% When X = I and Q in O(m), (Q^T X)_j = Q(:,j)^T (j-th col of Q transposed),
%% so phi_i = ||Q(:,2i-1)||^2 + ||Q(:,2i)||^2 = 1 + 1 = 2 (constant).
%% For any tangent vector W = Q*S with S skew-symmetric:
%%   delta(phi_i) = 2*Q(:,2i-1)^T*W(:,2i-1) + 2*Q(:,2i)^T*W(:,2i)
%%                = 2*S(2i-1,2i-1) + 2*S(2i,2i) = 0
%% since diagonal entries of skew-symmetric matrices vanish.
%% This eliminates all delta(phi) terms from the Hessian.
%% =========================================================================

function [A, Q, gval, history] = real_npp_identity(Y)
% Solves min_{A real normal} ||A - Y||_F^2  (real NPP with X = I).
    m = size(Y, 1);

    time_list     = [];
    cost_list     = [];
    gradnorm_list = [];
    t0 = tic;

    M = stiefelfactory(m, m);

    problem.M     = M;
    problem.cost  = @(Qvar)       -g_id(Qvar, Y);
    problem.egrad = @(Qvar)       -egrad_id(Qvar, Y);
    problem.ehess = @(Qvar, Wvar) -ehess_id(Qvar, Wvar, Y, M);

    options.statsfun  = @(problem, x, stats, store) record_stats(stats);
    options.verbosity = 0;
    options.Delta_bar = 1000;
    options.maxiter   = 2000;
    warning('off', 'manopt:getHessian:approx');

    Q0 = M.rand();
    [Q, neg_gval, ~, ~] = trustregions(problem, Q0, options);
    gval = -neg_gval;

    if isstruct(Q), Q = Q.Q; end

    history.time     = time_list - time_list(1);
    history.residual = norm(Y,'fro')^2 + cost_list;
    history.gradnorm = gradnorm_list;

    Delta = build_Delta_id(Q, Y);
    A     = Q * Delta * Q';

    function stats = record_stats(stats)
        time_list(end+1,1)     = toc(t0);
        cost_list(end+1,1)     = stats.cost;
        gradnorm_list(end+1,1) = stats.gradnorm;
    end
end

function val = g_id(Q, Y)
% g(Q) with X=I, phi_i=2 constant.
    QY  = Q' * Y;
    m   = size(QY, 1);
    val = 0;
    for i = 1:floor(m/2)
        QX_2i1 = Q(:,2*i-1)';
        QX_2i  = Q(:,2*i)';
        alpha_i = QX_2i1*QY(2*i-1,:)' + QX_2i*QY(2*i,:)';
        beta_i  = QX_2i*QY(2*i-1,:)'  - QX_2i1*QY(2*i,:)';
        val = val + (alpha_i^2 + beta_i^2) / 2;
    end
    if mod(m,2)==1
        val = val + (Q(:,m)'*QY(m,:)')^2 / 2;
    end
end

function Delta = build_Delta_id(Q, Y)
    QY    = Q' * Y;
    m     = size(QY, 1);
    Delta = zeros(m, m);
    for i = 1:floor(m/2)
        QX_2i1 = Q(:,2*i-1)';
        QX_2i  = Q(:,2*i)';
        a = (QX_2i1*QY(2*i-1,:)' + QX_2i*QY(2*i,:)') / 2;
        b = (QX_2i*QY(2*i-1,:)'  - QX_2i1*QY(2*i,:)') / 2;
        Delta(2*i-1:2*i, 2*i-1:2*i) = [a,b;-b,a];
    end
    if mod(m,2)==1
        Delta(m,m) = (Q(:,m)'*QY(m,:)') / 2;
    end
end

function G = egrad_id(Q, Y)
% Euclidean gradient with X=I:
%   nabla_Q g = 2(Y^T Q Delta* + Y Q Delta*^T - Q Delta*^T Delta*)
    Delta       = build_Delta_id(Q, Y);
    DeltaT      = Delta';
    G = 2 * (Y'*Q*Delta + Y*Q*DeltaT - Q*DeltaT*Delta);
end

function H = ehess_id(Q, W, Y, M)
% Euclidean Hessian with X=I, phi_i=2, delta(phi_i)=0.
% delta(a_i*) = d_alpha_i/2, delta(b_i*) = d_beta_i/2.
    W   = M.tangent2ambient(Q, W);
    QY  = Q' * Y;
    WY  = W' * Y;
    m   = size(QY, 1);

    Delta  = zeros(m, m);
    dDelta = zeros(m, m);

    for i = 1:floor(m/2)
        QX_2i1 = Q(:,2*i-1)';  QX_2i = Q(:,2*i)';
        WX_2i1 = W(:,2*i-1)';  WX_2i = W(:,2*i)';
        QY_2i1 = QY(2*i-1,:);  QY_2i = QY(2*i,:);
        WY_2i1 = WY(2*i-1,:);  WY_2i = WY(2*i,:);

        a = (QX_2i1*QY_2i1' + QX_2i*QY_2i') / 2;
        b = (QX_2i*QY_2i1'  - QX_2i1*QY_2i') / 2;
        Delta(2*i-1:2*i, 2*i-1:2*i) = [a,b;-b,a];

        % delta(phi_i) = 0, so delta(a*) = d_alpha/phi = d_alpha/2
        d_alpha = WX_2i1*QY_2i1' + WX_2i*QY_2i' + QX_2i1*WY_2i1' + QX_2i*WY_2i';
        d_beta  = WX_2i*QY_2i1'  - WX_2i1*QY_2i' + QX_2i*WY_2i1' - QX_2i1*WY_2i';
        dDelta(2*i-1:2*i, 2*i-1:2*i) = [d_alpha/2, d_beta/2; -d_beta/2, d_alpha/2];
    end

    if mod(m,2)==1
        QX_m = Q(:,m)';  WX_m = W(:,m)';
        QY_m = QY(m,:);  WY_m = WY(m,:);
        Delta(m,m)  = (QX_m*QY_m') / 2;
        dDelta(m,m) = (WX_m*QY_m' + QX_m*WY_m') / 2;
    end

    DeltaT       = Delta';
    dDeltaT      = dDelta';
    dDeltaTDelta = dDeltaT*Delta + DeltaT*dDelta;

    % X=I: XY^T = Y^T, YX^T = Y, XX^T = I
    H = 2 * ( Y'*W*Delta    + Y'*Q*dDelta    + ...
              Y*W*DeltaT     + Y*Q*dDeltaT    - ...
              W*DeltaT*Delta - Q*dDeltaTDelta );
end

%% =========================================================================
%% Guglielmi-Scalone Algorithm (real case, Omega = empty set)
%% Faithful implementation of Algorithm 1 + Section 3 from GS (2019).
%%
%% Functional (eq. 5):
%%   nu_eps(E) = (1/2)||A+eps*E||_F^2 - (1/2)*sum_k|lambda_k|^2
%%
%% Free gradient (eq. 16):
%%   G_f = A - sum_k (|lam_k|/|y_k'*x_k|) * y_k*x_k'
%% where ||x_k||=||y_k||=1 and y_k'*x_k = |y_k'*x_k|*exp(-i*theta_k)  (eq. 12)
%% For real case, Omega=empty (eq. 22): G = Re(G_f)
%%
%% Gradient system (eq. 23): dE/dt = -G + <G,E>_F * E
%% Euler with adaptive h (GS paper text):
%%   E_hat = E + h*(-G + <G,E>_F*E);  E_new = E_hat/||E_hat||_F
%%   if nu_eps(E_new) >= nu_eps(E): halve h and retry
%%
%% Outer secant (Algorithm 1): find eps* s.t. nu'(eps*)=0
%% where nu'(eps) = eps - ||G(eps)||_F  (Theorem 4)
%% =========================================================================

function [X_out, history] = guglielmi_scalone(A)
    m    = size(A,1);
    tol  = 1e-8;
    kmax = 30;

    time_list = [];
    res_list  = [];
    t0 = tic;

    % Step 1: initial gradient at E=0
    G0 = compute_gs_gradient(A, zeros(m,m), 0);
    if norm(G0,'fro') < 1e-14
        X_out = A; history.time = 0; history.residual = 0; return;
    end

    % Step 2: E0 = -G0/||G0||_F, initialize eps bounds
    E      = -G0 / norm(G0,'fro');
    eps_lb = 0;
    eps_ub = norm(A,'fro');
    eps_k  = eps_ub * 0.1;

    % Step 3: integrate
    E = integrate_gs(A, E, eps_k);
    G = compute_gs_gradient(A, E, eps_k);

    % Initial record
    time_list(end+1,1) = toc(t0);
    res_list(end+1,1)  = norm(A + eps_k*E - A,'fro')^2;  % = eps_k^2

    % Outer iteration: secant on nu'(eps) = eps - ||G(eps)||_F
    eps_hat = 0;
    nu_hat  = 0 - norm(G0,'fro');  % nu'(0)

    for k = 0:kmax-1
        nu_k = eps_k - norm(G,'fro');  % Theorem 4 / Step 8

        time_list(end+1,1) = toc(t0);
        res_list(end+1,1)  = norm(eps_k*E,'fro')^2;  % ||X - A||_F^2 = ||eps*E||_F^2

        if (eps_ub - eps_lb) < tol, break; end

        % Step 9: secant
        if abs(nu_k - nu_hat) > 1e-14
            eps_next = eps_k - (eps_k - eps_hat)/(nu_k - nu_hat) * nu_k;
        else
            eps_next = (eps_lb + eps_ub)/2;
        end

        % Step 10: update bounds
        if nu_k < -tol
            eps_hat = eps_lb;
            eps_lb  = max(eps_lb, eps_k);
        else
            eps_hat = eps_ub;
            eps_ub  = min(eps_ub, eps_k);
        end
        G_hat  = compute_gs_gradient(A, E, eps_hat);
        nu_hat = eps_hat - norm(G_hat,'fro');

        % Step 11: bisection fallback
        if eps_next <= eps_lb || eps_next >= eps_ub
            eps_next = (eps_lb + eps_ub)/2;
        end
        eps_k = eps_next;

        % Step 13: integrate
        E = integrate_gs(A, E, eps_k);
        G = compute_gs_gradient(A, E, eps_k);
    end

    X_out = A + eps_k * E;
    history.time     = time_list - time_list(1);
    history.residual = res_list;
end

function G = compute_gs_gradient(A, E, eps)
% Gradient for real case, Omega=empty (GS eq. 16 + eq. 22).
% G_f = A - sum_k (|lam_k|/|y_k'*x_k|) * y_k*x_k'
% For real A: G = Re(G_f).
% Note: for conjugate pairs, Re(yk*xk') + Re(conj(yk)*conj(xk)') = 2*Re(yk*xk'),
% so taking Re() at the end handles the pairing correctly without phase normalization.
    B      = A + eps * E;
    [V, D] = eig(B);
    lam    = diag(D);
    % Left eigenvectors: columns of inv(V)' (conjugate transpose of inv(V)).
    % Verified: if w = k-th col of inv(V)', then w^H*B = lambda_k*w^H.
    Y = inv(V)';

    Gf = A;
    for k = 1:length(lam)
        xk    = V(:,k) / norm(V(:,k));
        yk    = Y(:,k) / norm(Y(:,k));
        denom = abs(yk' * xk);
        if denom > 1e-14
            Gf = Gf - (abs(lam(k)) / denom) * real(yk * xk');
        end
    end
    G = real(Gf);
end

function nu = compute_nu_eps(A, E, eps)
% nu_eps(E) = (1/2)||A+eps*E||_F^2 - (1/2)*sum|lambda_k|^2  (GS eq. 5)
    B   = A + eps * E;
    lam = eig(B);
    nu  = 0.5*norm(B,'fro')^2 - 0.5*sum(abs(lam).^2);
end

function E = integrate_gs(A, E0, eps)
% Euler integration of dE/dt = -G + <G,E>_F*E with adaptive h.
% Adaptive rule: halve h if nu_eps does not decrease (GS paper text).
    E       = E0;
    h       = 0.1;
    tol     = 1e-9;
    nmax    = 3000;
    nu_curr = compute_nu_eps(A, E, eps);

    for iter = 1:nmax
        G  = compute_gs_gradient(A, E, eps);
        GE = sum(sum(G .* E));  % <G,E>_F for real matrices
        dE = -G + GE * E;

        % Adaptive step: halve h until functional decreases
        h_try = h;
        for bisect = 1:15
            E_hat = E + h_try * dE;
            nrm   = norm(E_hat,'fro');
            if nrm > 1e-14, E_new = E_hat / nrm; else, E_new = E; end
            nu_new = compute_nu_eps(A, E_new, eps);
            if nu_new < nu_curr + 1e-14, break; end
            h_try = h_try / 2;
        end
        E       = E_new;
        nu_curr = nu_new;
        h       = min(h_try * 1.05, 1.0);

        % Convergence check: stationary when E + G/||G|| ~ 0
        nG = norm(G,'fro');
        if nG > 1e-14 && norm(E + G/nG,'fro') < tol, break; end
        if nG < tol, break; end
    end
end