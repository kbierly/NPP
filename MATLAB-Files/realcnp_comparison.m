function realcnp_comparison()
% Compares two methods for the real closest normal matrix problem:
%   (1) our real NPP method with X = I_m  (Riemannian trust region on O(m))
%   (2) Guglielmi & Scalone's two-level gradient system (their real code)
%
% Runs on G&S Example 1 (3x3 real), on the real literature matrices B7, J7,
% T7, F12, and on random uniform real matrices at sizes 10, 20, 50, 100.
%
% Our method (X = I) is run from a random orthogonal start on SO(m); Guglielmi &
% Scalone's real gradient-flow method is run for comparison. Only G&S Example 1
% prints its recovered matrices; all other cases print scalar diagnostics only.
%
% Requires Manopt (https://www.manopt.org) on the path, and the
% Guglielmi-Scalone implementation; see README.

    rng(0);

    global GS_INNER_ITER_COUNT

    % ----- cases; 'show' prints recovered matrices (Example 1 only) -----
    cases = {};
    cases{end+1} = struct('name','G&S Example 1', 'Y', example1_matrix(), 'show', true);
    cases{end+1} = struct('name','Ruhe B7 (Gaussian)', 'Y', randn(7), 'show', false);
    cases{end+1} = struct('name','Ruhe J7 (Jordan block)', 'Y', jordan0(7), 'show', false);
    cases{end+1} = struct('name','Ruhe T7 (Nilpotent)', 'Y', triu(randn(7),1), 'show', false);
    cases{end+1} = struct('name','Ruhe F12 (Frank)', 'Y', frank(12), 'show', false);

    % ----- random uniform real -----
    for mm = [10 20 50 100]
        cases{end+1} = struct('name', 'Uniform real', ...
                              'Y', rand(mm), 'show', false);
    end

    for c = 1:numel(cases)
        run_case(cases{c});
    end
end

% =====================================================================
%  Run both methods on one matrix and report
% =====================================================================

function run_case(cs)
    Y = cs.Y;
    m = size(Y,1);
    nY = norm(Y, 'fro');

    fprintf('\n==================================================================\n');
    fprintf('%s   (m = %d)\n', cs.name, m);
    fprintf('==================================================================\n');
    fprintf('nu(Y) = %.4f,  ||Y||_F = %.4f\n', departure(Y), nY);

    % ---- our method (X = I), random start ----
    t0 = tic;
    [A1r, it1r, h1r] = solve_npp_xI(Y);
    tm1r = toc(t0);
    report('Ours (real NPP, random start)', Y, A1r, it1r, tm1r, h1r.residual(1));

    % ---- Guglielmi & Scalone (real) ----
    global GS_INNER_ITER_COUNT
    GS_INNER_ITER_COUNT = 0;
    t0 = tic;
    [A2, it2, h2] = gs_solve(Y);
    tm2 = toc(t0);
    gs_inner_iters = GS_INNER_ITER_COUNT;
    fprintf('  Guglielmi-Scalone (total inner iterations: %d)\n', gs_inner_iters);
    report('Guglielmi-Scalone', Y, A2, it2, tm2, h2.residual(1));

    if isfield(cs,'show') && cs.show
        fprintf('\n recovered normal matrix (ours):\n'); disp(A1r);
        fprintf(' recovered normal matrix (G&S):\n');  disp(A2);
    end

    plot_convergence(cs.name, m, h1r, h2);
end

% ---- residual-vs-time plot, both methods on a common time axis ----
function plot_convergence(name, m, h1r, h2)
    figure('Color','w','Name',sprintf('Convergence %s', name), ...
           'Units','inches','Position',[1 1 8 6]);
    ax = gca; hold(ax,'on');

    npp_rand  = [0.20 0.60 1.00];
    gs_col    = [0.85 0.20 0.55];

    rmin = inf; rmax = 0;
    [rmin,rmax] = addcurve(ax, h1r, '-', 'o', npp_rand,  rmin, rmax);
    [rmin,rmax] = addcurve(ax, h2,  '--', 's', gs_col,    rmin, rmax);
    hold(ax,'off');

    set(ax, 'TickLabelInterpreter', 'latex', 'FontSize', 21, 'LineWidth', 3, ...
        'XMinorTick', 'off', 'YMinorTick', 'off', 'Layer', 'top', ...
        'XScale', 'log', 'YScale', 'log');
    set_decade_xticks(ax, {h1r.time, h2.time});
    if isfinite(rmin) && rmax > 0
        ylim(ax, [10^(log10(rmin)-0.15), 10^(log10(rmax)+0.27)]);
    end
    set(ax, 'XMinorTick', 'off', 'YMinorTick', 'off');
    xlabel(ax, '\bf\it Time (s)', 'Interpreter', 'tex', 'FontSize', 24);
    ylabel(ax, '\bf\it ||A-Y||_F^2', 'Interpreter', 'tex', 'FontSize', 24);
    title(ax, sprintf('%s, m = %d', name, m), 'FontWeight', 'bold', 'FontSize', 24, 'Interpreter','none');
    box(ax, 'on');
    set(ax, 'Clipping', 'on');
    fname = regexprep(sprintf('%s_%dx%d', name, m, m), '[^a-zA-Z0-9]', '_');
    try
        exportgraphics(gcf, sprintf('realconv_%s.png', fname), 'Resolution', 300);
    catch
        saveas(gcf, sprintf('realconv_%s.png', fname));
    end
end

function [rmin,rmax] = addcurve(ax, h, style, marker, col, rmin, rmax)
% Plot one method's history; skip cleanly if degenerate. Marker on first point.
    t = h.time(:);
    r = h.residual(:);
    n = min(numel(t), numel(r));
    if n < 1, return; end
    t = t(1:n);  r = r(1:n);
    tpos = t(t > 0);
    if ~isempty(tpos)
        t(t <= 0) = min(tpos);
    else
        t = max(t, 1e-6);
    end
    r = max(r, 1e-16);
    rmin = min(rmin, min(r));  rmax = max(rmax, max(r));
    plot(ax, t, r, style, 'Color', col, 'LineWidth', 3, ...
         'Marker', marker, 'MarkerIndices', 1, 'MarkerSize', 14, ...
         'MarkerFaceColor', col);
end

function set_decade_xticks(ax, time_hist)
    tmin = inf; tmax = 0;
    for s = 1:numel(time_hist)
        tt = time_hist{s}(:);
        tpos = tt(tt > 0);
        if ~isempty(tpos)
            tmin = min(tmin, min(tpos));
            tmax = max(tmax, max(tpos));
        end
    end
    if ~isfinite(tmin) || tmax <= 0, return; end
    lo = floor(log10(tmin));
    hi = ceil(log10(tmax));
    e  = lo:hi;
    if numel(e) <= 3
        % few decades: add half-decade ticks so short-time cases show detail
        ticks = [];
        for k = lo:hi
            ticks = [ticks, 10^k, 3*10^k]; %#ok<AGROW>
        end
        ticks = ticks(ticks >= 10^lo & ticks <= 10^hi);
        xticks(ax, ticks);
        labs = cell(1,numel(ticks));
        for i = 1:numel(ticks)
            ee = log10(ticks(i));
            if abs(ee-round(ee)) < 1e-6
                labs{i} = sprintf('$10^{%d}$', round(ee));
            else
                labs{i} = '';
            end
        end
        xticklabels(ax, labs);
    else
        xticks(ax, 10.^e);
        xticklabels(ax, arrayfun(@(k) sprintf('$10^{%d}$', k), e, 'UniformOutput', false));
    end
    xlim(ax, [10^(log10(tmin) - 0.22), 10^hi]);
end

function report(label, Y, A, iters, tm, init_res)
    dif = A - Y;
    fro = norm(dif, 'fro');
    two = norm(dif, 2);
    nc  = norm(A*A' - A'*A, 'fro');
    nY  = norm(Y, 'fro');
    fprintf('\n  %s\n', label);
    if nargin >= 6 && ~isempty(init_res) && ~isnan(init_res)
        fprintf('    initial residual ||A-Y||_F^2: %.6f\n', init_res);
    end
    fprintf('    final residual ||A-Y||_F^2 : %.6f\n', fro^2);
    fprintf('    Frobenius distance ||A-Y||_F: %.6f\n', fro);
    fprintf('    2-norm distance   ||A-Y||_2 : %.6f\n', two);
    fprintf('    normality ||AA^T-A^TA||_F   : %.3e\n', nc);
    fprintf('    normalized dist d_N/||A||_F : %.4f\n', fro/nY);
    fprintf('    departure Delta_F/||A||_F   : %.4f\n', sqrt(max(departure(Y),0))/nY);
    if ~isnan(iters)
        fprintf('    iterations                  : %d\n', iters);
    end
    fprintf('    time (s)                    : %.4f\n', tm);
end

function nu = departure(Y)
    lam = eig(Y);
    nu = norm(Y,'fro')^2 - sum(abs(lam).^2);
end

% =====================================================================
%  Method 1: our real NPP with X = I_m
% =====================================================================

function [A, iters, hist] = solve_npp_xI(Y)
    m = size(Y,1);
    M = stiefelfactory(m, m);   % O(m)
    M.retr = M.retr_polar;      % second-order (polar) retraction
    problem.M     = M;
    problem.cost  = @(Q)    -npp_g(Q, Y);
    problem.egrad = @(Q)    -npp_grad(Q, Y);
    problem.ehess = @(Q, W) -npp_hess(Q, W, Y, M);

    options.verbosity   = 0;
    options.Delta_bar   = 1000;
    options.maxiter     = 2000;
    options.tolgradnorm = 1e-6;
    warning('off', 'manopt:getHessian:approx');

    tl = []; rl = [];
    t0 = tic;
    options.statsfun = @(p,Q,st,store) record(st);

    Q0 = M.rand();
    r0 = residual_at(Q0, Y);   % initial residual before any iteration

    [Q, ~] = trustregions(problem, Q0, options);
    A = recoverA(Q, Y);
    iters = numel(rl) - 1;

    hist.time     = [0; tl - tl(1)];
    hist.residual = [r0; norm(Y,'fro')^2 + rl];

    function st = record(st)
        tl(end+1,1) = toc(t0);
        rl(end+1,1) = st.cost;
    end
end


function r = residual_at(Q, Y)
    A = recoverA(Q, Y);
    r = norm(A - Y, 'fro')^2;
end

function A = recoverA(Q, Y)
    m = size(Y,1);
    D = optimal_quasidiagonal(Q, Y, m);
    A = Q*D*Q';
end

% ---------------------------------------------------------------------
%  Reduced X = I_m formulation. 
% ---------------------------------------------------------------------

% Optimal real quasidiagonal Delta^*(Q) and its per-block active form (X = I_m).
function [D, isrot] = optimal_quasidiagonal(Q, Y, m)
    B     = Q'*Y*Q;
    D     = zeros(m);
    nb    = floor(m/2);
    isrot = false(nb,1);

    for i = 1:nb
        r1 = 2*i-1;  r2 = 2*i;

        alpha = B(r1,r1) + B(r2,r2);
        beta  = B(r1,r2) - B(r2,r1);

        rot_score = (alpha^2 + beta^2) / 2;
        dia_score = B(r1,r1)^2 + B(r2,r2)^2;

        if rot_score >= dia_score
            a = alpha / 2;   b = beta / 2;
            D(r1,r1) =  a;  D(r1,r2) = b;
            D(r2,r1) = -b;  D(r2,r2) = a;
            isrot(i) = true;
        else
            D(r1,r1) = B(r1,r1);
            D(r2,r2) = B(r2,r2);
        end
    end

    if mod(m,2) == 1
        D(m,m) = B(m,m);
    end
end

% Directional derivative delta(Delta^*) along the ambient direction W (X = I_m).
function dD = delta_quasidiagonal(Q, W, Y, m)
    dB = W'*Y*Q + Q'*Y*W;

    [~, isrot] = optimal_quasidiagonal(Q, Y, m);

    dD = zeros(m);
    for i = 1:floor(m/2)
        r1 = 2*i-1;  r2 = 2*i;
        if isrot(i)
            da = (dB(r1,r1) + dB(r2,r2)) / 2;
            db = (dB(r1,r2) - dB(r2,r1)) / 2;
            dD(r1,r1) =  da;  dD(r1,r2) = db;
            dD(r2,r1) = -db;  dD(r2,r2) = da;
        else
            dD(r1,r1) = dB(r1,r1);
            dD(r2,r2) = dB(r2,r2);
        end
    end

    if mod(m,2) == 1
        dD(m,m) = dB(m,m);
    end
end

% g(Q) = sum of the per-block maximal scores.
function val = npp_g(Q, Y)
    m = size(Y,1);
    B = Q'*Y*Q;
    val = 0;
    for i = 1:floor(m/2)
        r1 = 2*i-1;  r2 = 2*i;
        alpha = B(r1,r1) + B(r2,r2);
        beta  = B(r1,r2) - B(r2,r1);
        val = val + max( (alpha^2 + beta^2)/2, B(r1,r1)^2 + B(r2,r2)^2 );
    end
    if mod(m,2) == 1
        val = val + B(m,m)^2;
    end
end

function G = npp_grad(Q, Y)
    m = size(Y,1);
    D = optimal_quasidiagonal(Q, Y, m);
    % X = I_m in  2( X Y^T Q D + Y X^T Q D^T - X X^T Q D^T D )
    G = 2*( (Y')*Q*D + Y*Q*D' - Q*(D')*D );
end

function H = npp_hess(Q, W, Y, M)
    m = size(Y,1);
    W  = M.tangent2ambient(Q, W);
    D  = optimal_quasidiagonal(Q, Y, m);
    dD = delta_quasidiagonal(Q, W, Y, m);

    H = 2*( (Y')*W*D  + (Y')*Q*dD ...
          + Y*W*D' + Y*Q*dD' ...
          - W*(D')*D - Q*((dD')*D + (D')*dD) );
end

% =====================================================================
%  Method 2: Guglielmi & Scalone
%
%  Redacted. Requires the Guglielmi-Scalone implementation, see README.
% =====================================================================

function [B, k, hist] = gs_solve(Y)
    error('Requires the Guglielmi-Scalone implementation; see README.');
end

% =====================================================================
%  Test matrices
% =====================================================================

function A = jordan0(n)
    A = diag(ones(n-1,1), 1);            % Jordan block, zero eigenvalue
end

function A = frank(n)
    A = zeros(n);
    for j = 1:n
        for k = 1:n
            if k >= j-1, A(j,k) = min(j,k); end
        end
    end
end

function A = example1_matrix()
    A = [ 0.3  -1     0;
          1     0.5  -0.3;
          0    -1     0 ];
end
