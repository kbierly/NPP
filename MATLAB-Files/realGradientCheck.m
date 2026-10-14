% Numerically verify the Euclidean gradient formula for the real
% Normal Procrustes Problem (NPP) using Manopt's checkgradient.
%
% Objective (to maximize):
%   g(Q) = sum_i max{ (alpha_i^2 + beta_i^2)/(phi_{2i-1} + phi_{2i}),
%                     gamma_{2i-1}^2/phi_{2i-1} + gamma_{2i}^2/phi_{2i} }
%          (+ gamma_m^2/phi_m if m is odd),
% where phi_j = ||(Q^T X)_j||^2, gamma_j = (Q^T Y)_j (Q^T X)_j^T,
%   alpha_i = (Q^T X)_{2i-1}(Q^T Y)_{2i-1}^T + (Q^T X)_{2i}(Q^T Y)_{2i}^T,
%   beta_i  = (Q^T X)_{2i}(Q^T Y)_{2i-1}^T - (Q^T X)_{2i-1}(Q^T Y)_{2i}^T,
% and the optimal block Delta_i^* is the rotation form [a b; -b a] when the
% first term attains the max, and the diagonal form diag(l_{2i-1}, l_{2i})
% otherwise.
%
% Manopt *minimizes*, so we pass cost = -g and egrad = -grad g.
%
% Runs the check on four problem sizes: (m,n) = (50,50), (50,40), (50,25), (50,10).
% checkgradient draws the first-order Taylor error E(t) on a log-log plot
% and reports the slope, and we display the slope in the figure title.
%
% Requires Manopt on the path (https://www.manopt.org).
function realGradientCheck()
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
        M.retr = M.retr_polar;   % second-order (polar) retraction
        problem.M     = M;
        problem.cost  = @(Q) costfun(Q, X, Y, m);
        problem.egrad = @(Q) egradfun(Q, X, Y, m);

        figure('Color', 'w', 'Name', sprintf('m=%d, n=%d', m, n), ...
               'Units', 'inches', 'Position', [1 1 8 6]);

        out   = evalc('checkgradient(problem);');
        tok   = regexp(out, 'It appears to be:\s*<strong>([\d.]+)</strong>', ...
                       'tokens', 'once');
        slope = str2double(tok{1});

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

        set(ax, 'TickLabelInterpreter', 'latex', 'FontSize', 21, 'LineWidth', 3, 'XMinorTick', 'off', 'YMinorTick', 'off', 'Layer', 'top');

        yticks(ax, [1e-15, 1e-10, 1e-5, 1]);

        xlabel(ax, '\bf\it t', 'Interpreter', 'tex', 'FontSize', 24);
        ylabel(ax, '\bf\it E(t)', 'Interpreter', 'tex', 'FontSize', 24);
        title(ax, sprintf('m = %d, n = %d, Slope = %.5f', m, n, slope), 'FontWeight', 'bold', 'FontSize', 24);
        text(3e-8, 3e-2, {'Dashed line:','Slope = 2'}, 'FontSize', 18);
        fprintf('(m,n) = (%d,%d): slope = %.5f\n', m, n, slope);

        pngFileName = sprintf('RealGradientCheck_m%d_n%d.png', m, n);
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