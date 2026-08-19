function cnp_comparison()
% Compares three methods for the complex closest normal matrix problem:
%   (1) our NPP method with X = I_m  (Riemannian trust region on U(m))
%   (2) Guglielmi & Scalone's two-level gradient system (complex-adapted)
%   (3) Ruhe's Jacobi algorithm (Algorithm J)
%
% Runs on the literature matrices (G&S Examples 1, 8, 9; Ruhe 2xF2, B7, J7,
% T7, F12) and on random uniform complex matrices at sizes 10, 20, 50, 100, 200.
% Our method is run from a random start. For each case
% prints initial/final residual, Frobenius and 2-norm distance, normality,
% iterations (ours, G&S) or sweeps (Ruhe), time, and Ruhe's normalized distance
% and departure. Recovered matrices shown for small cases. Convergence
% (residual vs wall-clock time) is plotted per case.
%
% Requires Manopt (https://www.manopt.org) on the path.

    rng(0);

    global GS_INNER_ITER_COUNT

    % ----- literature matrices -----
    cases = {};
    cases{end+1} = struct('name','G&S Example 1', 'Y', example1_matrix(), 'show', true);
    cases{end+1} = struct('name','Ruhe 2x2 / G&S Example 8', 'Y', ruhe2x2_matrix(), 'show', true);
    cases{end+1} = struct('name','G&S Example 9', 'Y', example9_matrix(), 'show', false);
    cases{end+1} = struct('name','Ruhe B7 (Gaussian)', 'Y', randn(7)+1i*randn(7), 'show', false);
    cases{end+1} = struct('name','Ruhe J7 (Jordan block)', 'Y', jordan_block(7), 'show', false);
    cases{end+1} = struct('name','Ruhe T7 (Nilpotent)', 'Y', nilpotent(7), 'show', false);
    cases{end+1} = struct('name','Ruhe F12 (Frank)', 'Y', frank_matrix(12), 'show', false);

    % ----- random uniform complex -----
    for mm = [10 20 50 100 200 250]
        cases{end+1} = struct('name', 'Uniform complex', ...
                              'Y', rand(mm)+1i*rand(mm), 'show', true);
    end

    for c = 1:numel(cases)
        run_case(cases{c});
    end
end

% =====================================================================
%  Run all three methods on one matrix and report
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
    report('Ours (NPP, random start)', Y, A1r, it1r, NaN, tm1r, h1r.residual(1));

    % ---- Guglielmi & Scalone (complex) ----
    global GS_INNER_ITER_COUNT
    GS_INNER_ITER_COUNT = 0;
    t0 = tic;
    [A2, it2, h2] = gs_solve(Y);
    tm2 = toc(t0);
    gs_inner_iters = GS_INNER_ITER_COUNT;
    fprintf('  Guglielmi-Scalone (total inner iterations: %d)\n', gs_inner_iters);
    report('Guglielmi-Scalone', Y, A2, it2, NaN, tm2, h2.residual(1));

    % ---- Ruhe ----
    t0 = tic;
    [A3, sw3, h3] = ruhe_solve(Y);
    tm3 = toc(t0);
    report('Ruhe (Algorithm J)', Y, A3, NaN, sw3, tm3, h3.residual(1));

    if cs.show
        fprintf('\n recovered normal matrix (ours):\n'); disp(A1r);
        fprintf(' recovered normal matrix (G&S):\n');  disp(A2);
        fprintf(' recovered normal matrix (Ruhe):\n'); disp(A3);
    end

    plot_convergence(cs.name, m, h1r, h2, h3);
end

% ---- residual-vs-time plot, all methods on a common time axis ----
function plot_convergence(name, m, h1r, h2, h3)
    figure('Color','w','Name',sprintf('Convergence %s', name), ...
           'Units','inches','Position',[1 1 8 6]);
    ax = gca; hold(ax,'on');

    npp_rand  = [0.20 0.60 1.00];
    gs_col    = [0.85 0.20 0.55];
    ruhe_col  = [0.10 0.65 0.35];

    rmin = inf; rmax = 0;
    [rmin,rmax] = addcurve(ax, h1r, '-', 'o', npp_rand,  rmin, rmax);
    [rmin,rmax] = addcurve(ax, h2,  '--', 's', gs_col,    rmin, rmax);
    [rmin,rmax] = addcurve(ax, h3,  ':',  '^', ruhe_col,  rmin, rmax);
    hold(ax,'off');

    set(ax, 'TickLabelInterpreter', 'latex', 'FontSize', 21, 'LineWidth', 3, ...
        'XMinorTick', 'off', 'YMinorTick', 'off', 'Layer', 'top', ...
        'XScale', 'log', 'YScale', 'log');
    set_decade_xticks(ax, {h1r.time, h2.time, h3.time});
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
        exportgraphics(gcf, sprintf('conv_%s.png', fname), 'Resolution', 300);
    catch
        saveas(gcf, sprintf('conv_%s.png', fname));
    end
    close(gcf);
end

function [rmin,rmax] = addcurve(ax, h, style, marker, col, rmin, rmax)
% Plot one method's history; skip cleanly if degenerate. Marker on first point.
    t = h.time(:);
    r = h.residual(:);
    n = min(numel(t), numel(r));
    if n < 1, return; end
    t = t(1:n);  r = r(1:n);
    % shift a zero start to the smallest positive time so it sits in-range on a log axis
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

function report(label, Y, A, iters, sweeps, tm, init_res)
    dif = A - Y;
    fro = norm(dif, 'fro');
    two = norm(dif, 2);
    nc  = norm(A*A' - A'*A, 'fro');
    nY  = norm(Y, 'fro');
    fprintf('\n  %s\n', label);
    if nargin >= 7 && ~isempty(init_res) && ~isnan(init_res)
        fprintf('    initial residual ||A-Y||_F^2: %.6f\n', init_res);
    end
    fprintf('    final residual ||A-Y||_F^2 : %.6f\n', fro^2);
    fprintf('    Frobenius distance ||A-Y||_F: %.6f\n', fro);
    fprintf('    2-norm distance   ||A-Y||_2 : %.6f\n', two);
    fprintf('    normality ||AA*-A*A||_F     : %.3e\n', nc);
    fprintf('    normalized dist d_N/||A||_F : %.4f\n', fro/nY);
    fprintf('    departure Delta_F/||A||_F   : %.4f\n', sqrt(max(departure(Y),0))/nY);
    if ~isnan(iters)
        fprintf('    iterations                  : %d\n', iters);
    end
    if ~isnan(sweeps)
        fprintf('    sweeps                      : %d\n', sweeps);
    end
    fprintf('    time (s)                    : %.4f\n', tm);
end

function nu = departure(Y)
    lam = eig(Y);
    nu = norm(Y,'fro')^2 - sum(abs(lam).^2);
end

% =====================================================================
%  Method 1: our NPP with X = I_m
% =====================================================================

function [A, iters, hist] = solve_npp_xI(Y)
    m = size(Y,1);
    M = unitaryfactory(m);
    problem.M     = M;
    problem.cost  = @(U)    -npp_f(U, Y);
    problem.egrad = @(U)    -npp_grad(U, Y);
    problem.ehess = @(U, W) -npp_hess(U, W, Y, M);

    options.verbosity   = 0;
    options.Delta_bar   = 1000;
    options.maxiter     = 2000;
    options.tolgradnorm = 1e-6;
    warning('off', 'manopt:getHessian:approx');

    tl = []; rl = [];
    t0 = tic;
    options.statsfun = @(p,U,st,store) record(st);

    U0 = M.rand();

    r0 = residual_at(U0, Y);   % initial residual before any iteration

    [U, ~] = trustregions(problem, U0, options);
    if isstruct(U), U = U.U; end
    A = recoverA(U, Y);
    iters = numel(rl) - 1;

    hist.time     = [0; tl - tl(1)];
    hist.residual = [r0; norm(Y,'fro')^2 + rl];   % prepend initial residual at t=0

    function st = record(st)
        tl(end+1,1) = toc(t0);
        rl(end+1,1) = st.cost;
    end
end

function r = residual_at(U, Y)
    A = recoverA(U, Y);
    r = norm(A - Y, 'fro')^2;
end

function A = recoverA(U, Y)
    if isstruct(U), U = U.U; end
    A = U * diag(diag(U'*Y*U)) * U';
end

function val = npp_f(U, Y)
    if isstruct(U), U = U.U; end
    val = sum(abs(diag(U'*Y*U)).^2);
end

function G = npp_grad(U, Y)
    if isstruct(U), U = U.U; end
    d = diag(U'*Y*U);
    G = 2*(Y'*U*diag(d) + Y*U*diag(conj(d)) - U*diag(abs(d).^2));
end

function H = npp_hess(U, W, Y, M)
    if isstruct(U), U = U.U; end
    W  = M.tangent2ambient(U, W);
    d  = diag(U'*Y*U);
    dd = diag(W'*Y*U + U'*Y*W);
    D   = diag(d);   Dc = diag(conj(d));   D2 = diag(abs(d).^2);
    dD  = diag(dd);  dDc = diag(conj(dd));
    dD2 = dD*Dc + D*dDc;
    H = 2*( Y'*W*D + Y'*U*dD + Y*W*Dc + Y*U*dDc - W*D2 - U*dD2 );
end

% =====================================================================
%  Method 2: Guglielmi & Scalone (complex-adapted port of their code; 
% we modify only eps1 to accommodate larger eps* at the sizes we test)
% =====================================================================

function [B, k, hist] = gs_solve(Y)
    n   = size(Y,1);
    tol = 1e-16;
    eps0 = 0.01;
    eps1 = max(9, norm(Y,'fro'));   % widen bracket for large eps*
    [B, k, hist] = secanti(eps0, eps1, Y, n, tol);
end

function [B, k, hist] = secanti(eps0, eps1, A, n, tol)
    hmin = 1e-4;
    nmax = 10000;
    E0   = zeros(n, n);
    epsv = 0.1;
    k    = 1;
    eps_lb = eps0;
    eps_ub = eps1;
    eps_hat = eps_lb;
    B = A;
    tl = []; rl = [];
    t0 = tic;
    while abs(eps_lb - eps_ub) > 1e-7
        [~, ~, B, ~, df]     = normality_com(A, n, hmin, tol, epsv(k), nmax, E0);
        der = df;
        tl(end+1,1) = toc(t0);
        rl(end+1,1) = norm(B - A, 'fro')^2;   % ||A+eps*E - A||_F^2
        if norm(B - A, 'fro') > 1e-12
            E0 = (B - A) / norm(B - A, 'fro');
        else
            E0 = randn(n) / norm(randn(n), 'fro');
        end
        [~, ~, ~, ~, df_hat] = normality_com(A, n, hmin, tol, eps_hat, nmax, E0);
        if abs(der - df_hat) > 1e-15
            coeff = (epsv(k) - eps_hat) / (der - df_hat);
            epsv(k+1) = epsv(k) - coeff * der;
        else
            epsv(k+1) = (eps_lb + eps_ub) / 2;
        end
        if der < -4e-7
            eps_hat = eps_lb;  eps_lb = max(eps_lb, epsv(k));
        else
            eps_hat = eps_ub;  eps_ub = min(eps_ub, epsv(k));
        end
        if epsv(k+1) > eps_ub || epsv(k+1) < eps_lb
            epsv(k+1) = (eps_lb + eps_ub) / 2;
        end
        k = k + 1;
        if k > 100, break; end
    end
    if isempty(tl)
        hist.time = 0; hist.residual = norm(B-A,'fro')^2;
    else
        hist.time = tl - tl(1); hist.residual = rl;
    end
end

function [f1, iter, B, G, df] = normality_com(A, n, hmin, tol, eps, nmax, E0)
    P = ones(n, n);      % empty pattern (plain CNP)
    iter = 0;  h = 3;  hmax = 5;  G = zeros(n,n);  df = 0;
    if norm(E0, 'fro') > 0
        E0 = E0 / norm(E0, 'fro');
    else
        E0 = randn(n);  E0 = E0 / norm(E0, 'fro');
    end
    B = A + eps * (P .* E0);
    f0 = departure(B);
    if abs(f0) < tol || eps < 1e-13
        G = P .* gradient_G(A, B, n);
        df = 2 * (eps - norm(G, 'fro'));
        f1 = f0;  return;
    end
    while f0 > tol
        iter = iter + 1;
        B = A + eps * (P .* E0);
        G = P .* gradient_G(A, B, n);
        df = 2 * (eps - norm(G, 'fro'));
        Edot = -P .* (G + real(trace(E0' * G)) * E0);   % complex: no real() on Edot
        E1 = P .* (E0 + h * Edot);
        E1 = E1 / norm(E1, 'fro');
        Bnew = A + eps * E1;
        f1 = departure(Bnew);
        if f1 < f0
            if abs(f0 - f1) < 1e-14, B = Bnew; return; end
            if h < hmax, h = h * 1.2; end
            E0 = E1;  f0 = f1;  B = Bnew;
        else
            if h < hmin, f1 = f0; return; end
            h = h / 2;
        end
        if iter > nmax, f1 = f0; return; end
    end
    f1 = f0;
end

function G = gradient_G(A, B, n)
    global GS_INNER_ITER_COUNT
    GS_INNER_ITER_COUNT = GS_INNER_ITER_COUNT + 1;
    [R, D, L] = eig(B);
    Lambda = diag(D);
    g_sum = zeros(n, n);
    for j = 1:n
        y = L(:, j) / norm(L(:, j));
        x = R(:, j) / norm(R(:, j));
        theta = angle(Lambda(j));
        beta  = angle(y' * x);
        alpha = -theta - beta;
        x_rot = x * exp(1i * alpha);
        r = abs(y' * x_rot);
        g_sum = g_sum + (abs(Lambda(j)) / r) * y * x_rot';
    end
    G = A - g_sum;      % complex: no real()
end

% =====================================================================
%  Method 3: Ruhe's Algorithm J  (validated vs his printed 2x2, J7, F12)
%  theta = 0.5*arg(-det(block)),  h = e^{-i th} a_jk + e^{i th} conj(a_kj),
%  denom = Re(e^{-i th}(a_jj - a_kk)),  phi = 0.5*atan2(|h|, denom).
% =====================================================================

function [N, sweeps, hist] = ruhe_solve(A)
    n = size(A,1);
    Acur = A;
    Z = eye(n);
    max_sweeps = 100000;
    tol = 1e-9;          % Ruhe's cutoff: max pivot magnitude |h_jk| < 1e-9 (Table 1)
    sweeps = 0;
    tl = []; rl = [];
    t0 = tic;
    tl(end+1,1) = toc(t0);            % initial point (before any sweep)
    rl(end+1,1) = offdiag_mass(Acur); % = ||Y - diag(Y)||_F^2
    for sweep = 1:max_sweeps
        max_h = 0;
        for j = 1:n
            for k = j+1:n
                ajj = Acur(j,j);  akk = Acur(k,k);
                ajk = Acur(j,k);  akj = Acur(k,j);
                mm = (ajj + akk) / 2;
                detblock = (ajj - mm)*(akk - mm) - ajk*akj;
                theta = 0.5 * angle(-detblock);
                hjk = exp(-1i*theta)*ajk + exp(1i*theta)*conj(akj);
                if abs(hjk) > max_h, max_h = abs(hjk); end
                denom = real(exp(-1i*theta) * (ajj - akk));
                phi = 0.5 * atan2(abs(hjk), denom);
                if abs(hjk) > 0, alpha = angle(hjk); else, alpha = 0; end
                c = cos(phi);  s = sin(phi);
                ejp = exp(1i*alpha);  ejm = exp(-1i*alpha);
                % Apply Acur <- R' * Acur * R with R = I except 2x2 block at (j,k):
                %   R = [c, -ejp*s; ejm*s, c].  Update only columns j,k then rows j,k.
                cj = Acur(:,j);  ck = Acur(:,k);
                Acur(:,j) = cj*c        + ck*(ejm*s);
                Acur(:,k) = cj*(-ejp*s) + ck*c;
                rj = Acur(j,:);  rk = Acur(k,:);
                Acur(j,:) = c*rj        + (ejp*s)*rk;
                Acur(k,:) = (-ejm*s)*rj + c*rk;
                zj = Z(:,j);  zk = Z(:,k);
                Z(:,j) = zj*c        + zk*(ejm*s);
                Z(:,k) = zj*(-ejp*s) + zk*c;
            end
        end
        sweeps = sweep;
        res = offdiag_mass(Acur);   % ||N-A||_F^2 if diagonalized now
        tl(end+1,1) = toc(t0);
        rl(end+1,1) = res;
        if max_h < tol, break; end  % Ruhe's 1e-9 cutoff on max pivot magnitude
    end
    N = Z * diag(diag(Acur)) * Z';
    if isempty(tl)
        hist.time = 0; hist.residual = offdiag_mass(Acur);
    else
        hist.time = tl - tl(1); hist.residual = rl;
    end
end

function s = offdiag_mass(A)
    s = norm(A - diag(diag(A)), 'fro')^2;
end

% =====================================================================
%  Test matrices
% =====================================================================

function A = example1_matrix()
    A = [ 0.3  -1     0;
          1     0.5  -0.3;
          0    -1     0 ];
end

function A = ruhe2x2_matrix()
    A = [ 0.7616+1.2296i, -1.4740-0.4577i;
         -1.6290-2.6378i,  0.1885-0.8575i ];
end

function A = example9_matrix()
    A = [ ...
       13i,      -17+44i,   40+31i,   -73+34i,  -17+19i,   26+31i,   44+44i ; ...
       42-10i,   -29-25i,   -74+4i,   -19+11i,  -23+29i,   -35+9i,   -69-23i ; ...
      -32+50i,    10-19i,    8+36i,   -60-39i,   35+0i,   -22+85i,   42-69i ; ...
       -6-5i,    -35-40i,   -11+22i,    6-5i,    50+64i,   61-50i,   -19-28i ; ...
       28-22i,   -23-27i,    50-81i,   35+24i,    5+19i,  -38+44i,   -14+8i ; ...
       79+23i,   -51-68i,    31+61i,    2-44i,   -43-9i,    8-15i,    62+18i ; ...
      -25-49i,   -17-30i,   -53+30i,  -24+60i,   25-77i,  -64+10i,   27-24i ];
end

function A = jordan_block(n)
    A = diag(ones(n-1,1), 1);
    A = complex(A);
end

function A = nilpotent(n)
    R = randn(n) + 1i * randn(n);
    A = triu(R, 1);
end

function A = frank_matrix(n)
    A = zeros(n);
    for i = 1:n
        for j = 1:n
            if j >= i-1
                A(i,j) = n + 1 - max(i,j);
            end
        end
    end
    A = complex(A);
end