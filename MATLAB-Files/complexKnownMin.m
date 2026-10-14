% Known global optimum recovery for the complex Normal Procrustes Problem (NPP).
%
% For each size (m,n) we plant a known zero-residual solution: a random normal
% matrix A* = U_true D U_true^*, a random X, and Y = A* X. By construction a
% normal matrix achieving residual zero exists. We feed only X and Y to the
% solver and test whether it recovers A* (or, when n < m, some other normal
% matrix achieving the same optimal residual).
%
% The solver is run from two random unitary starts. For each start we report the
% final residual and the difference between the recovered matrix and A* (absolute
% and relative to ||A*||_F), along with the spread between the two final
% residuals. Residual and gradient norm trajectories are plotted (one figure
% each) per case.
%
% Requires Manopt on the path (https://www.manopt.org).

function complexKnownMin()

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

        M = unitaryfactory(m);

        % Plant a known normal solution A* = U_true D U_true^*, with the unitary
        % factor drawn from the Haar measure (M.rand) and uniform eigenvalues
        U_true = M.rand();
        d_true = (2*rand(m,1)-1) + 1i*(2*rand(m,1)-1);
        A_true = U_true * diag(d_true) * U_true';

        % Uniform X, and Y defined so that A* X - Y = 0 exactly
        X = rand(m,n) + 1i*rand(m,n);
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

% ===== Solver =========================================================

function [A, res, init_res, hist] = solve_npp(M, X, Y, m, U0)
    problem.M     = M;
    problem.cost  = @(U) costfun(U, X, Y, m);
    problem.egrad = @(U) egradfun(U, X, Y, m);
    problem.ehess = @(U, V) ehessfun(U, V, X, Y, m, M);

    % Initial residual at the starting point U0
    init_res = residual_at(U0, X, Y, m);

    t0 = tic;
    tl = []; rl = []; gl = [];
    options.statsfun  = @(prob, U, stats, store) record(stats);
    options.verbosity = 0;
    options.Delta_bar = 1000;
    options.maxiter   = 2000;
    warning('off', 'manopt:getHessian:approx');

    U = trustregions(problem, U0, options);

    hist.time     = tl - tl(1);
    hist.residual = norm(Y,'fro')^2 + rl;
    hist.gradnorm = gl;

    UX = U'*X;  UY = U'*Y;
    phi = sum(abs(UX).^2, 2);
    d = zeros(m,1);
    nz = phi > 0;
    d(nz) = sum(UY(nz,:) .* conj(UX(nz,:)), 2) ./ phi(nz);
    A = U*diag(d)*U';

    res = norm(A*X - Y, 'fro')^2;

    function stats = record(stats)
        tl(end+1,1) = toc(t0);
        rl(end+1,1) = stats.cost;
        gl(end+1,1) = stats.gradnorm;
    end
end

function r = residual_at(U, X, Y, m)
    UX = U'*X;  UY = U'*Y;
    phi = sum(abs(UX).^2, 2);
    d = zeros(m,1);
    nz = phi > 0;
    d(nz) = sum(UY(nz,:) .* conj(UX(nz,:)), 2) ./ phi(nz);
    A = U*diag(d)*U';
    r = norm(A*X - Y, 'fro')^2;
end

% ===== Plotting =======================================================

function plot_residual(res_hist, time_hist, colors, styles, markers, m, n)
    figure('Color', 'w', 'Name', sprintf('KnownMin Residual m=%d n=%d', m, n), ...
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
    exportgraphics(gcf, sprintf('KnownMin_Residual_m%d_n%d.png', m, n), 'Resolution', 300);
end

function plot_gradnorm(grad_hist, time_hist, colors, styles, markers, m, n)
    figure('Color', 'w', 'Name', sprintf('KnownMin Gradnorm m=%d n=%d', m, n), ...
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
    exportgraphics(gcf, sprintf('KnownMin_GradNorm_m%d_n%d.png', m, n), 'Resolution', 300);
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

function [d, phi] = optimal_diagonal(U, X, Y, m)
    UX  = U'*X;
    UY  = U'*Y;
    phi = sum(abs(UX).^2, 2);
    gamma = sum(UY .* conj(UX), 2);
    d = zeros(m,1);
    nz = phi > 0;
    d(nz) = gamma(nz) ./ phi(nz);
end

function val = costfun(U, X, Y, m)
    [d, phi] = optimal_diagonal(U, X, Y, m);
    nz = phi > 0;
    val = -sum( abs(d(nz)).^2 .* phi(nz) );
end

function G = egradfun(U, X, Y, m)
    [d, ~] = optimal_diagonal(U, X, Y, m);
    D  = diag(d);
    Dc = diag(conj(d));
    D2 = diag(abs(d).^2);
    gradf = 2*( X*(Y')*U*D + Y*(X')*U*Dc - X*(X')*U*D2 );
    G = -gradf;
end

function H = ehessfun(U, V, X, Y, m, M)
    V  = M.tangent2ambient(U, V);
    UX = U'*X;  UY = U'*Y;
    VX = V'*X;  VY = V'*Y;
    phi = sum(abs(UX).^2, 2);

    gamma = sum(UY .* conj(UX), 2);
    d = zeros(m,1);
    nz = phi > 0;
    d(nz) = gamma(nz) ./ phi(nz);

    dgamma = sum(VY .* conj(UX), 2) + sum(UY .* conj(VX), 2);
    dphi   = 2*real( sum(UX .* conj(VX), 2) );
    dd = zeros(m,1);
    dd(nz) = dgamma(nz)./phi(nz) - d(nz).*dphi(nz)./phi(nz);

    D   = diag(d);
    Dc  = diag(conj(d));
    D2  = diag(abs(d).^2);
    dD  = diag(dd);
    dDc = diag(conj(dd));
    dD2 = dD*Dc + D*dDc;

    hessf = 2*( X*(Y')*V*D  + X*(Y')*U*dD ...
              + Y*(X')*V*Dc + Y*(X')*U*dDc ...
              - X*(X')*V*D2 - X*(X')*U*dD2 );
    H = -hessf;
end