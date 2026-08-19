% Numerically verify the Euclidean Hessian formula for the real
% Normal Procrustes Problem (NPP) using Manopt's checkhessian.
%
% Objective (to maximize):
%   g(Q) = sum_i max{ (alpha_i^2 + beta_i^2)/(phi_{2i-1} + phi_{2i}),
%                     gamma_{2i-1}^2/phi_{2i-1} + gamma_{2i}^2/phi_{2i} }
%          (+ gamma_m^2/phi_m if m is odd),
% with the optimal block Delta_i^* the rotation form [a b; -b a] when the first
% term attains the max, and the diagonal form diag(l_{2i-1}, l_{2i}) otherwise.
%
% Manopt *minimizes*, so we pass cost = -g, egrad = -grad g, ehess = -Hess g.
%
% Runs the check on four problem sizes: (m,n) = (50,50), (50,40), (50,25), (50,10).
% checkhessian draws the second-order Taylor error E(t) on a log-log plot
% and reports the slope (which should be 3), and we display it in the title.
% stiefelfactory's polar retraction (M.retr_polar, set below) is second order, so
% checkhessian is valid at a random point; no critical point is required.
%
% Requires Manopt on the path (https://www.manopt.org).

function realHessianCheck()

    sizes = [50 50;
             50 40;
             50 25;
             50 10];

    for k = 1:size(sizes,1)
        m = sizes(k,1);
        n = sizes(k,2);
        rng(0);

        X = randn(m,n);
        Y = randn(m,n);

        M = stiefelfactory(m, m);
        M.retr = M.retr_polar;   % second-order (polar) retraction for a valid Hessian check
        problem.M     = M;
        problem.cost  = @(Q) costfun(Q, X, Y, m);
        problem.egrad = @(Q) egradfun(Q, X, Y, m);
        problem.ehess = @(Q, W) ehessfun(Q, W, X, Y, m, problem.M);

        % Adjust styling
        figure('Color', 'w', 'Name', sprintf('m=%d, n=%d', m, n), ...
               'Units', 'inches', 'Position', [1 1 8 6]);

        % Parse for slope and residuals
        out   = evalc('checkhessian(problem);');
        tok   = regexp(out, 'It appears to be:\s*<strong>([\d.]+)</strong>', ...
                       'tokens', 'once');
        slope = str2double(tok{1});

        tanTok = regexp(out, 'zero up to machine precision:\s*<strong>([\d.eE+-]+)</strong>', ...
                        'tokens', 'once');
        r_tan = str2double(tanTok{1});

        linTok = regexp(out, 'Value:\s*<strong>([\d.eE+-]+)</strong>', ...
                        'tokens', 'once');
        r_lin = str2double(linTok{1});

        symTok = regexp(out, '=\s*<strong>([\d.eE+-]+)</strong>', 'tokens');
        r_sym = str2double(symTok{end}{1});

        ax  = gca;
        fig = gcf;
        fig.Color = 'w';

        allLines = findobj(gca, 'Type', 'line');
        for lIdx = 1:length(allLines)
            if strcmp(allLines(lIdx).LineStyle, '--')
                allLines(lIdx).LineWidth = 3;
            elseif allLines(lIdx).LineWidth == 3
                allLines(lIdx).LineWidth = 8;
            else
                allLines(lIdx).LineWidth = 3;
                xData = allLines(lIdx).XData;
                yData = allLines(lIdx).YData;
                validIdx = yData > 0;
                allLines(lIdx).XData = xData(validIdx);
                allLines(lIdx).YData = yData(validIdx);
            end
        end

        set(ax, 'TickLabelInterpreter', 'latex', 'FontSize', 21, 'LineWidth', 3, ...
            'XMinorTick', 'off', 'YMinorTick', 'off', 'Layer', 'top');
        yticks(ax, [1e-15, 1e-10, 1e-5, 1]);
        xlabel(ax, '\bf\it t', 'Interpreter', 'tex', 'FontSize', 24);
        ylabel(ax, '\bf\it E_H(t)', 'Interpreter', 'tex', 'FontSize', 24);
        title(ax, {sprintf('          m = %d, n = %d, Slope = %.5f', m, n, slope), ...
                   sprintf('r_{tan} = %.1e,  r_{lin} = %.1e,  r_{sym} = %.1e', ...
                           r_tan, r_lin, r_sym)}, ...
              'FontWeight', 'bold', 'FontSize', 20);
        text(3e-8, 3e-2, {'Dashed line:','Slope = 3'}, 'FontSize', 18);

        fprintf('%s', out);

        pngFileName = sprintf('RealHessianCheck_m%d_n%d.png', m, n);
        exportgraphics(fig, pngFileName, 'Resolution', 300);
    end
end

% ===== Local functions (take X, Y, m explicitly) =======================

% Optimal real quasidiagonal Delta^*(Q) and its per-block active form.
function [D, isrot] = optimal_quasidiagonal(Q, X, Y, m)
    QX = Q'*X;                              % j-th row is (Q^T X)_j
    QY = Q'*Y;
    phi   = sum(QX.^2, 2);                  % phi_j = ||(Q^T X)_j||^2
    gamma = sum(QY .* QX, 2);               % gamma_j = (Q^T Y)_j (Q^T X)_j^T

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
% assembled block-wise according to the active form of each block:
%   rotation form: delta(a_i) = (delta(alpha_i) - a_i*(delta(phi_{2i-1})+delta(phi_{2i})))/(phi_{2i-1}+phi_{2i}),
%                  delta(b_i) likewise with beta_i;
%   diagonal form: delta(l_j)  = (delta(gamma_j) - l_j*delta(phi_j))/phi_j,
% where delta(phi_j) = 2 (Q^T X)_j (W^T X)_j^T and
%       delta(gamma_j) = (W^T Y)_j (Q^T X)_j^T + (Q^T Y)_j (W^T X)_j^T.
function dD = delta_quasidiagonal(Q, W, X, Y, m)
    QX = Q'*X;  QY = Q'*Y;
    WX = W'*X;  WY = W'*Y;
    phi    = sum(QX.^2, 2);
    gamma  = sum(QY .* QX, 2);
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
    val = norm(D*(Q'*X) - Q'*Y, 'fro')^2 - norm(Y, 'fro')^2;   % minimize -g
end

function G = egradfun(Q, X, Y, m)
    D = optimal_quasidiagonal(Q, X, Y, m);
    gradg = 2*( X*(Y')*Q*D + Y*(X')*Q*D' - X*(X')*Q*(D')*D );
    G = -gradg;
end

function H = ehessfun(Q, W, X, Y, m, M)
    W  = M.tangent2ambient(Q, W);          % represent the direction in the ambient space
    D  = optimal_quasidiagonal(Q, X, Y, m);
    dD = delta_quasidiagonal(Q, W, X, Y, m);

    hessg = 2*( X*(Y')*W*D  + X*(Y')*Q*dD ...
              + Y*(X')*W*D' + Y*(X')*Q*dD' ...
              - X*(X')*W*(D')*D - X*(X')*Q*((dD')*D + (D')*dD) );
    H = -hessg;              % Hessian of the minimized cost -g
end