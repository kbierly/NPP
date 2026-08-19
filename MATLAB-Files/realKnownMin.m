% Known global optimum recovery for the real Normal Procrustes Problem (NPP).
%
% For each size (m,n) we plant a known zero-residual solution: a random real
% normal matrix A* = Q_true Delta_true Q_true^T, a random X, and Y = A* X. By
% construction a real normal matrix achieving residual zero exists. We feed only
% X and Y to the solver and test whether it recovers A* (or, when n < m, some
% other real normal matrix achieving the same optimal residual).
%
% The solver is run from two random orthogonal starts. For each start we report
% the final residual and the difference between the recovered matrix and A*
% (absolute and relative to ||A*||_F), along with the spread between the two
% final residuals. Residual and gradient norm trajectories are plotted (one
% figure each) per case.
%
% Requires Manopt on the path (https://www.manopt.org).

function realKnownMin()

    sizes = [10 2; 10 5; 10 8; 10 10;
             20 4; 20 10; 20 16; 20 20;
             50 10; 50 25; 50 40; 50 50];
    rng(0);

    % Start 1: blue solid circle. Start 2: red dashed square.
    start_colors  = [0.20 0.60 1.00;
                     0.85 0.10 0.10];
    start_styles  = {'-','--'};
    start_markers = {'o','s'};

    fprintf(['%-4s %-4s | %-11s %-11s | %-11s %-11s | %-11s %-11s | ', ...
             '%-11s %-11s | %-11s %-11s | %-11s\n'], ...
        'm', 'n', 'init(r1)', 'init(r2)', 'res(r1)', 'res(r2)', ...
        'time(r1)', 'time(r2)', 'dA(r1)', 'dA(r2)', ...
        'relA(r1)%', 'relA(r2)%', 'res spread');
    fprintf('%s\n', repmat('-', 1, 160));

    for k = 1:size(sizes,1)
        m = sizes(k,1);
        n = sizes(k,2);

        M = stiefelfactory(m, m);   % O(m); WLOG restrict to one component (see paper)
        M.retr = M.retr_polar;      % second-order (polar) retraction for a valid Hessian check

        % Plant a known real normal solution A* = Q_true Delta_true Q_true^T, with
        % the orthogonal factor drawn from the Haar measure (M.rand) and a uniform
        % quasidiagonal spectrum
        Q_true = M.rand();
        A_true = Q_true * random_quasidiagonal(m) * Q_true';

        % Uniform real X, and Y defined so that A* X - Y = 0 exactly
        X = rand(m,n);
        Y = A_true * X;

        normAt   = norm(A_true, 'fro');
        init_s   = zeros(2,1);
        res_s    = zeros(2,1);
        time_s   = zeros(2,1);
        adiff_s  = zeros(2,1);
        reladiff = zeros(2,1);
        res_hist  = cell(2,1);
        grad_hist = cell(2,1);
        time_hist = cell(2,1);

        for s = 1:2
            t0 = tic;
            [A, res, init_res, hist] = solve_npp(M, X, Y, m, M.rand());
            time_s(s)    = toc(t0);
            init_s(s)    = init_res;
            res_s(s)     = res;
            adiff_s(s)   = norm(A - A_true, 'fro');
            reladiff(s)  = adiff_s(s) / normAt * 100;
            res_hist{s}  = hist.residual;
            grad_hist{s} = hist.gradnorm;
            time_hist{s} = hist.time;
        end

        plot_residual(res_hist, time_hist, start_colors, start_styles, start_markers, m, n);
        plot_gradnorm(grad_hist, time_hist, start_colors, start_styles, start_markers, m, n);

        spread = abs(res_s(1) - res_s(2));

        fprintf(['%-4d %-4d | %-11.3e %-11.3e | %-11.3e %-11.3e | ', ...
                 '%-11.3f %-11.3f | %-11.4f %-11.4f | %-11.4f %-11.4f | %-11.3e\n'], ...
            m, n, init_s(1), init_s(2), res_s(1), res_s(2), ...
            time_s(1), time_s(2), adiff_s(1), adiff_s(2), ...
            reladiff(1), reladiff(2), spread);
    end
end

% A random real quasidiagonal matrix: each 2x2 block is either a rotation form
% [a b; -b a] or a diagonal form diag(l1,l2), plus a 1x1 block if m is odd.
function Delta = random_quasidiagonal(m)
    Delta = zeros(m);
    for i = 1:floor(m/2)
        r1 = 2*i-1;  r2 = 2*i;
        if rand() < 0.5
            a = 2*rand()-1;  b = 2*rand()-1;
            Delta(r1,r1) =  a;  Delta(r1,r2) = b;
            Delta(r2,r1) = -b;  Delta(r2,r2) = a;
        else
            Delta(r1,r1) = 2*rand()-1;
            Delta(r2,r2) = 2*rand()-1;
        end
    end
    if mod(m,2) == 1
        Delta(m,m) = 2*rand()-1;
    end
end


% ===== Solver =========================================================

function [A, res, init_res, hist] = solve_npp(M, X, Y, m, Q0)
    problem.M     = M;
    problem.cost  = @(Q) costfun(Q, X, Y, m);
    problem.egrad = @(Q) egradfun(Q, X, Y, m);
    problem.ehess = @(Q, W) ehessfun(Q, W, X, Y, m, M);

    % Initial residual at the starting point Q0
    init_res = residual_at(Q0, X, Y, m);

    t0 = tic;
    tl = []; rl = []; gl = [];
    options.statsfun  = @(prob, Q, stats, store) record(stats);
    options.verbosity = 0;
    options.Delta_bar = 1000;
    options.maxiter   = 2000;
    warning('off', 'manopt:getHessian:approx');

    Q = trustregions(problem, Q0, options);

    hist.time     = tl - tl(1);
    hist.residual = norm(Y,'fro')^2 + rl;
    hist.gradnorm = gl;

    A = recoverA(Q, X, Y, m);
    res = norm(A*X - Y, 'fro')^2;

    function stats = record(stats)
        tl(end+1,1) = toc(t0);
        rl(end+1,1) = stats.cost;
        gl(end+1,1) = stats.gradnorm;
    end
end

function r = residual_at(Q, X, Y, m)
    A = recoverA(Q, X, Y, m);
    r = norm(A*X - Y, 'fro')^2;
end

function A = recoverA(Q, X, Y, m)
    D = optimal_quasidiagonal(Q, X, Y, m);
    A = Q*D*Q';
end

% ===== Plotting =======================================================

function plot_residual(res_hist, time_hist, colors, styles, markers, m, n)
    figure('Color', 'w', 'Name', sprintf('RealKnownMin Residual m=%d n=%d', m, n), ...
           'Units', 'inches', 'Position', [1 1 8 6]);
    ax = gca; hold(ax, 'on');
    for s = 1:numel(res_hist)
        mi = find(time_hist{s} > 0, 1);
        semilogy(time_hist{s}, max(res_hist{s}, 1e-16), styles{s}, 'Color', colors(s,:), ...
                 'LineWidth', 3, 'Marker', markers{s}, 'MarkerIndices', mi, 'MarkerSize', 12, ...
                 'MarkerFaceColor', colors(s,:));
    end
    hold(ax, 'off');
    ymax = 0;
    for s = 1:numel(res_hist)
        ymax = max(ymax, max(res_hist{s}));
    end
    set(ax, 'TickLabelInterpreter', 'latex', 'FontSize', 21, 'LineWidth', 3, ...
        'XMinorTick', 'off', 'YMinorTick', 'off', 'Layer', 'top', 'YScale', 'log', 'XScale', 'log');
    ylim(ax, [ax.YLim(1), ymax*10^0.61]);
    set_decade_xticks(ax, time_hist);
    set(ax, 'XMinorTick', 'off', 'YMinorTick', 'off');
    xlabel(ax, '\bf\it Time (s)', 'Interpreter', 'tex', 'FontSize', 24);
    ylabel(ax, '\bf\it ||AX-Y||_F^2', 'Interpreter', 'tex', 'FontSize', 24);
    title(ax, sprintf('m = %d, n = %d', m, n), 'FontWeight', 'bold', 'FontSize', 24);
    box(ax, 'on');
    set(ax, 'Clipping', 'on');
    exportgraphics(gcf, sprintf('RealKnownMin_Residual_m%d_n%d.png', m, n), 'Resolution', 300);
end

function plot_gradnorm(grad_hist, time_hist, colors, styles, markers, m, n)
    figure('Color', 'w', 'Name', sprintf('RealKnownMin Gradnorm m=%d n=%d', m, n), ...
           'Units', 'inches', 'Position', [1 1 8 6]);
    ax = gca; hold(ax, 'on');
    for s = 1:numel(grad_hist)
        mi = find(time_hist{s} > 0, 1);
        semilogy(time_hist{s}, max(grad_hist{s}, 1e-16), styles{s}, 'Color', colors(s,:), ...
                 'LineWidth', 3, 'Marker', markers{s}, 'MarkerIndices', mi, 'MarkerSize', 12, ...
                 'MarkerFaceColor', colors(s,:));
    end
    hold(ax, 'off');
    ymax = 0;
    for s = 1:numel(grad_hist)
        ymax = max(ymax, max(grad_hist{s}));
    end
    set(ax, 'TickLabelInterpreter', 'latex', 'FontSize', 21, 'LineWidth', 3, ...
        'XMinorTick', 'off', 'YMinorTick', 'off', 'Layer', 'top', 'YScale', 'log', 'XScale', 'log');
    ylim(ax, [ax.YLim(1), ymax*10^0.61]);
    set_decade_xticks(ax, time_hist);
    set(ax, 'XMinorTick', 'off', 'YMinorTick', 'off');
    xlabel(ax, '\bf\it Time (s)', 'Interpreter', 'tex', 'FontSize', 24);
    ylabel(ax, '\bf\it grad norm', 'Interpreter', 'tex', 'FontSize', 24);
    title(ax, sprintf('m = %d, n = %d', m, n), 'FontWeight', 'bold', 'FontSize', 24);
    box(ax, 'on');
    set(ax, 'Clipping', 'on');
    exportgraphics(gcf, sprintf('RealKnownMin_GradNorm_m%d_n%d.png', m, n), 'Resolution', 300);
end

function set_decade_xticks(ax, time_hist)
    tmin = inf; tmax = 0;
    for s = 1:numel(time_hist)
        tpos = time_hist{s}(time_hist{s} > 0);
        if ~isempty(tpos)
            tmin = min(tmin, min(tpos));
            tmax = max(tmax, max(tpos));
        end
    end
    lo = floor(log10(tmin));
    hi = ceil(log10(tmax));
    e  = lo:hi;
    xticks(ax, 10.^e);
    xticklabels(ax, arrayfun(@(k) sprintf('$10^{%d}$', k), e, 'UniformOutput', false));
    xlim(ax, [10^(log10(tmin) - 0.22), 10^hi]);
end

% ===== Objective, gradient, Hessian ===================================
%
% Objective (to maximize):
%   g(Q) = sum_i max{ (alpha_i^2+beta_i^2)/(phi_{2i-1}+phi_{2i}),
%                     gamma_{2i-1}^2/phi_{2i-1} + gamma_{2i}^2/phi_{2i} }
%          (+ gamma_m^2/phi_m if m is odd),
% with the optimal block Delta_i^* the rotation form [a b; -b a] when the first
% term attains the max, and the diagonal form diag(l_{2i-1}, l_{2i}) otherwise.

% Optimal real quasidiagonal Delta^*(Q) and its per-block active form.
function [D, isrot] = optimal_quasidiagonal(Q, X, Y, m)
    QX = Q'*X;  QY = Q'*Y;
    phi   = sum(QX.^2, 2);              % phi_j = ||(Q^T X)_j||^2
    gamma = sum(QY .* QX, 2);           % gamma_j = (Q^T Y)_j (Q^T X)_j^T

    D     = zeros(m);
    nb    = floor(m/2);
    isrot = false(nb,1);

    for i = 1:nb
        r1 = 2*i-1;  r2 = 2*i;
        Phi = phi(r1) + phi(r2);

        alpha = QX(r1,:)*QY(r1,:)' + QX(r2,:)*QY(r2,:)';
        beta  = QX(r2,:)*QY(r1,:)' - QX(r1,:)*QY(r2,:)';

        if Phi > 0
            rot_score = (alpha^2 + beta^2) / Phi;
        else
            rot_score = 0;
        end

        dia_score = 0;
        if phi(r1) > 0, dia_score = dia_score + gamma(r1)^2 / phi(r1); end
        if phi(r2) > 0, dia_score = dia_score + gamma(r2)^2 / phi(r2); end

        if rot_score >= dia_score && Phi > 0
            a = alpha / Phi;   b = beta / Phi;
            D(r1,r1) =  a;  D(r1,r2) = b;
            D(r2,r1) = -b;  D(r2,r2) = a;
            isrot(i) = true;
        else
            if phi(r1) > 0, D(r1,r1) = gamma(r1) / phi(r1); end
            if phi(r2) > 0, D(r2,r2) = gamma(r2) / phi(r2); end
        end
    end

    if mod(m,2) == 1 && phi(m) > 0
        D(m,m) = gamma(m) / phi(m);
    end
end

% Directional derivative delta(Delta^*) along the ambient direction W,
% assembled block-wise according to the active form of each block.
function dD = delta_quasidiagonal(Q, W, X, Y, m)
    QX = Q'*X;  QY = Q'*Y;
    WX = W'*X;  WY = W'*Y;
    phi   = sum(QX.^2, 2);
    gamma = sum(QY .* QX, 2);
    dphi   = 2 * sum(QX .* WX, 2);
    dgamma = sum(WY .* QX, 2) + sum(QY .* WX, 2);

    [~, isrot] = optimal_quasidiagonal(Q, X, Y, m);

    dD = zeros(m);
    for i = 1:floor(m/2)
        r1 = 2*i-1;  r2 = 2*i;
        Phi = phi(r1) + phi(r2);
        if isrot(i)
            if Phi > 0
                alpha = QX(r1,:)*QY(r1,:)' + QX(r2,:)*QY(r2,:)';
                beta  = QX(r2,:)*QY(r1,:)' - QX(r1,:)*QY(r2,:)';
                a = alpha / Phi;   b = beta / Phi;

                dalpha = WX(r1,:)*QY(r1,:)' + QX(r1,:)*WY(r1,:)' ...
                       + WX(r2,:)*QY(r2,:)' + QX(r2,:)*WY(r2,:)';
                dbeta  = WX(r2,:)*QY(r1,:)' + QX(r2,:)*WY(r1,:)' ...
                       - WX(r1,:)*QY(r2,:)' - QX(r1,:)*WY(r2,:)';
                dPhi   = dphi(r1) + dphi(r2);

                da = (dalpha - a*dPhi) / Phi;
                db = (dbeta  - b*dPhi) / Phi;
                dD(r1,r1) =  da;  dD(r1,r2) = db;
                dD(r2,r1) = -db;  dD(r2,r2) = da;
            end
        else
            for r = [r1 r2]
                if phi(r) > 0
                    lam = gamma(r) / phi(r);
                    dD(r,r) = (dgamma(r) - lam*dphi(r)) / phi(r);
                end
            end
        end
    end

    if mod(m,2) == 1 && phi(m) > 0
        lam = gamma(m) / phi(m);
        dD(m,m) = (dgamma(m) - lam*dphi(m)) / phi(m);
    end
end

function val = costfun(Q, X, Y, m)
    D = optimal_quasidiagonal(Q, X, Y, m);
    % g(Q) = ||Y||_F^2 - ||Delta^* Q^T X - Q^T Y||_F^2; minimize -g
    val = norm(D*(Q'*X) - Q'*Y, 'fro')^2 - norm(Y, 'fro')^2;
end

function G = egradfun(Q, X, Y, m)
    D = optimal_quasidiagonal(Q, X, Y, m);
    gradg = 2*( X*(Y')*Q*D + Y*(X')*Q*D' - X*(X')*Q*(D')*D );
    G = -gradg;
end

function H = ehessfun(Q, W, X, Y, m, M)
    W  = M.tangent2ambient(Q, W);
    D  = optimal_quasidiagonal(Q, X, Y, m);
    dD = delta_quasidiagonal(Q, W, X, Y, m);

    hessg = 2*( X*(Y')*W*D  + X*(Y')*Q*dD ...
              + Y*(X')*W*D' + Y*(X')*Q*dD' ...
              - X*(X')*W*(D')*D - X*(X')*Q*((dD')*D + (D')*dD) );
    H = -hessg;
end