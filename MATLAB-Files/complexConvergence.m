% Evaluate convergence of the Riemannian trust-region solver on the complex
% Normal Procrustes Problem (NPP).
%
% Objective (to maximize):
%   f(U) = sum_i |gamma_i|^2 / phi_i,
%   gamma_i = (U^*Y)_i (U^*X)_i^*,  phi_i = ||(U^*X)_i||^2,
% with optimal diagonal d_i^* = gamma_i / phi_i.
%
% For each size (m,n) and each input distribution, X and Y are fixed and the
% solver is run from three random unitary starting points. Residual and gradient
% norm trajectories are plotted (one figure each), and the best-of-three-random
% results are reported in a table.
%
% Requires Manopt on the path (https://www.manopt.org).

function complexConvergence()

    sizes  = [10 2; 10 5; 10 8; 10 10;
              20 4; 20 10; 20 16; 20 20;
              50 10; 50 25; 50 40; 50 50];
    dists  = {'Gaussian', 'Uniform'};
    nstart = 3;                              % 3 random
    rng(0);

    start_colors = [0.20 0.60 1.00;
                    0.85 0.10 0.10;
                    0.20 0.80 0.20];

    start_styles  = {'-','--',':'};
    start_markers = {'o','s','^'};
    start_labels  = {'Random 1','Random 2','Random 3'};

    for di = 1:numel(dists)
        dist = dists{di};

        best_init = zeros(size(sizes,1), 1);
        best_res  = zeros(size(sizes,1), 1);
        best_time = zeros(size(sizes,1), 1);
        best_grad = zeros(size(sizes,1), 1);
        best_nerr = zeros(size(sizes,1), 1);

        for k = 1:size(sizes,1)
            m = sizes(k,1);
            n = sizes(k,2);

            if strcmp(dist, 'Gaussian')
                X = randn(m,n) + 1i*randn(m,n);
                Y = randn(m,n) + 1i*randn(m,n);
            else
                X = rand(m,n) + 1i*rand(m,n);
                Y = rand(m,n) + 1i*rand(m,n);
            end

            M = unitaryfactory(m);

            save(sprintf('XY_%s_m%d_n%d.mat', dist, m, n), 'X', 'Y');

            res_hist  = cell(nstart,1);
            grad_hist = cell(nstart,1);
            time_hist = cell(nstart,1);
            init_val  = zeros(nstart,1);
            res_end   = zeros(nstart,1);
            time_end  = zeros(nstart,1);
            grad_end  = zeros(nstart,1);
            nerr_end  = zeros(nstart,1);
            A_all     = cell(nstart,1);

            for s = 1:nstart
                U0 = M.rand();
                [A, hist] = solve_npp(M, X, Y, m, U0);
                A_all{s}     = A;
                res_hist{s}  = hist.residual;
                grad_hist{s} = hist.gradnorm;
                time_hist{s} = hist.time;
                init_val(s)  = hist.residual(1);
                res_end(s)   = hist.residual(end);
                time_end(s)  = hist.time(end);
                grad_end(s)  = hist.gradnorm(end);
                nerr_end(s)  = norm(A*A' - A'*A, 'fro');
                fprintf('  %s  m=%d n=%d  start %d/%d done (res=%.4f, time=%.2fs)\n', ...
                        dist, m, n, s, nstart, res_end(s), time_end(s));
            end

            maxAdiff = 0;
            for i = 1:nstart
                for j = i+1:nstart
                    maxAdiff = max(maxAdiff, norm(A_all{i} - A_all{j}, 'fro'));
                end
            end
            maxAnorm = 0;
            for i = 1:nstart
                maxAnorm = max(maxAnorm, norm(A_all{i}, 'fro'));
            end
            relAdiff = maxAdiff / maxAnorm * 100;
            fprintf('  %s  m=%d n=%d  max pairwise ||A_i - A_j||_F = %.3e  (rel = %.4f%%)\n', ...
                    dist, m, n, maxAdiff, relAdiff);

            [best_res(k), bi] = min(res_end);
            best_init(k) = init_val(bi);
            best_time(k) = time_end(bi);
            best_grad(k) = grad_end(bi);
            best_nerr(k) = nerr_end(bi);

            plot_residual(res_hist, time_hist, start_colors, start_styles, start_markers, start_labels, m, n, dist);
            plot_gradnorm(grad_hist, time_hist, start_colors, start_styles, start_markers, start_labels, m, n, dist);
        end

        print_table(sizes, best_init, best_res, best_time, best_grad, best_nerr, ...
                    sprintf('Best of three random starts (%s inputs)', dist));
    end
end

% ===== Solver =========================================================

function [A, hist] = solve_npp(M, X, Y, m, U0)
    problem.M     = M;
    problem.cost  = @(U) costfun(U, X, Y, m);
    problem.egrad = @(U) egradfun(U, X, Y, m);
    problem.ehess = @(U, V) ehessfun(U, V, X, Y, m, M);

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

    function stats = record(stats)
        tl(end+1,1) = toc(t0);
        rl(end+1,1) = stats.cost;
        gl(end+1,1) = stats.gradnorm;
    end
end

% ===== Plotting =======================================================

function plot_residual(res_hist, time_hist, colors, styles, markers, labels, m, n, dist)
    figure('Color', 'w', 'Name', sprintf('Residual m=%d n=%d %s', m, n, dist), ...
           'Units', 'inches', 'Position', [1 1 8 6]);
    ax = gca; hold(ax, 'on');
    for s = 1:numel(res_hist)
        mi = find(time_hist{s} > 0, 1);
        plot(time_hist{s}, res_hist{s}, styles{s}, 'Color', colors(s,:), 'LineWidth', 3, ...
             'Marker', markers{s}, 'MarkerIndices', mi, 'MarkerSize', 14, ...
             'MarkerFaceColor', colors(s,:));
    end
    hold(ax, 'off');
    ymax = 0;
    for s = 1:numel(res_hist)
        ymax = max(ymax, max(res_hist{s}));
    end
    set(ax, 'TickLabelInterpreter', 'latex', 'FontSize', 21, 'LineWidth', 3, ...
        'XMinorTick', 'off', 'YMinorTick', 'off', 'Layer', 'top', 'XScale', 'log');
    ylim(ax, [0, ymax*1.08]);
    set_decade_xticks(ax, time_hist);
    set(ax, 'XMinorTick', 'off');
    xlabel(ax, '\bf\it Time (s)', 'Interpreter', 'tex', 'FontSize', 24);
    ylabel(ax, '\bf\it ||AX-Y||_F^2', 'Interpreter', 'tex', 'FontSize', 24);
    title(ax, sprintf('%s, m = %d, n = %d', dist, m, n), 'FontWeight', 'bold', 'FontSize', 24);
    box(ax, 'on');
    set(ax, 'Clipping', 'on');
    exportgraphics(gcf, sprintf('Convergence_%s_m%d_n%d.png', dist, m, n), 'Resolution', 300);
end

function plot_gradnorm(grad_hist, time_hist, colors, styles, markers, labels, m, n, dist)
    figure('Color', 'w', 'Name', sprintf('Gradnorm m=%d n=%d %s', m, n, dist), ...
           'Units', 'inches', 'Position', [1 1 8 6]);
    ax = gca; hold(ax, 'on');
    for s = 1:numel(grad_hist)
        mi = find(time_hist{s} > 0, 1);
        semilogy(time_hist{s}, max(grad_hist{s}, 1e-16), styles{s}, 'Color', colors(s,:), ...
                 'LineWidth', 3, 'Marker', markers{s}, 'MarkerIndices', mi, 'MarkerSize', 14, ...
                 'MarkerFaceColor', colors(s,:));
    end
    hold(ax, 'off');
    ymax = 0;
    for s = 1:numel(grad_hist)
        ymax = max(ymax, max(grad_hist{s}));
    end
    set(ax, 'TickLabelInterpreter', 'latex', 'FontSize', 21, 'LineWidth', 3, ...
        'XMinorTick', 'off', 'YMinorTick', 'off', 'Layer', 'top', 'YScale', 'log', 'XScale', 'log');
    ylim(ax, [ax.YLim(1), ymax*10^0.22]);
    set_decade_xticks(ax, time_hist);
    set(ax, 'XMinorTick', 'off');
    xlabel(ax, '\bf\it Time (s)', 'Interpreter', 'tex', 'FontSize', 24);
    ylabel(ax, '\bf\it grad norm', 'Interpreter', 'tex', 'FontSize', 24);
    title(ax, sprintf('%s, m = %d, n = %d', dist, m, n), 'FontWeight', 'bold', 'FontSize', 24);
    box(ax, 'on');
    set(ax, 'Clipping', 'on');
    exportgraphics(gcf, sprintf('GradNorm_%s_m%d_n%d.png', dist, m, n), 'Resolution', 300);
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

% ===== Tables =========================================================

function print_table(sizes, init, res, time, grad, nerr, header)
    fprintf('\n%s\n', header);
    fprintf('%-6s %-6s %-14s %-14s %-12s %-12s %-12s\n', ...
        'm', 'n', 'Init res', 'Residual', 'Time(s)', 'Grad norm', 'Norm err');
    fprintf('%s\n', repmat('-', 1, 80));
    for k = 1:size(sizes,1)
        fprintf('%-6d %-6d %-14.4f %-14.4f %-12.3f %-12.2e %-12.2e\n', ...
            sizes(k,1), sizes(k,2), init(k), res(k), time(k), grad(k), nerr(k));
    end
end

% ===== Objective, gradient, Hessian ===================================

function [d, phi] = optimal_diagonal(U, X, Y, m)
    UX  = U'*X;                            % i-th row is (U^*X)_i
    UY  = U'*Y;
    phi = sum(abs(UX).^2, 2);              % phi_i = ||(U^*X)_i||^2
    gamma = sum(UY .* conj(UX), 2);        % gamma_i = (U^*Y)_i (U^*X)_i^*
    d = zeros(m,1);
    nz = phi > 0;
    d(nz) = gamma(nz) ./ phi(nz);          % d_i^* = gamma_i / phi_i
end

function val = costfun(U, X, Y, m)
    [d, phi] = optimal_diagonal(U, X, Y, m);
    nz = phi > 0;
    val = -sum( abs(d(nz)).^2 .* phi(nz) );   % minimize -f
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