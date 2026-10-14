% Numerically verify the Euclidean Hessian formula for the complex
% Normal Procrustes Problem (NPP) using Manopt's checkhessian.
%
% Objective (to maximize):
%   f(U) = sum_i |gamma_i|^2 / phi_i,
%   gamma_i = (U^*Y)_i (U^*X)_i^*,  phi_i = ||(U^*X)_i||^2,
% with optimal diagonal d_i^* = gamma_i / phi_i.
%
% Manopt *minimizes*, so we pass cost = -f, egrad = -grad f, ehess = -Hess f.
%
% Runs the check on four problem sizes: (m,n) = (50,50), (50,40), (50,25), (50,10).
% checkhessian draws the second-order Taylor error E(t) on a log-log plot
% and reports the slope (which should be 3), and we display it in the title.
% unitaryfactory provides M.exp (a second-order retraction), so the check is
% valid at a random point; no critical point is required.
%
% Requires Manopt on the path (https://www.manopt.org).

function complexHessianCheck()

    sizes = [50 50;
             50 40;
             50 25;
             50 10];

    for k = 1:size(sizes,1)
        m = sizes(k,1);
        n = sizes(k,2);
        rng(0);

        X = randn(m,n) + 1i*randn(m,n);
        Y = randn(m,n) + 1i*randn(m,n);

        problem.M     = unitaryfactory(m);
        problem.cost  = @(U) costfun(U, X, Y, m);
        problem.egrad = @(U) egradfun(U, X, Y, m);
        problem.ehess = @(U, V) ehessfun(U, V, X, Y, m, problem.M);

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

        pngFileName = sprintf('HessianCheck_m%d_n%d.png', m, n);
        exportgraphics(fig, pngFileName, 'Resolution', 300);
    end
end

% ===== Local functions (take X, Y, m explicitly) =======================

function [d, phi] = optimal_diagonal(U, X, Y, m)
    UX  = U'*X;                            % i-th row is (U^*X)_i
    UY  = U'*Y;
    phi = sum(abs(UX).^2, 2);              % phi_i = ||(U^*X)_i||^2
    gamma = sum(UY .* conj(UX), 2);        % gamma_i = (U^*Y)_i (U^*X)_i^*
    d = zeros(m,1);                        % initialize all zeros on the diagonal
    nz = phi > 0;
    d(nz) = gamma(nz) ./ phi(nz);          % if phi_i nonzero then d_i^* = gamma_i / phi_i
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
%   dd_i = (dgamma_i)/phi_i - d_i (dphi_i)/phi_i,
%   dgamma_i = (V^*Y)_i (U^*X)_i^* + (U^*Y)_i (V^*X)_i^*,
%   dphi_i   = 2 Re[(U^*X)_i (V^*X)_i^*]
    V  = M.tangent2ambient(U, V);          % represent the direction in the ambient space
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
    dD2 = dD*Dc + D*dDc;     % delta(|D|^2)

    hessf = 2*( X*(Y')*V*D  + X*(Y')*U*dD ...
              + Y*(X')*V*Dc + Y*(X')*U*dDc ...
              - X*(X')*V*D2 - X*(X')*U*dD2 );
    H = -hessf;              % Hessian of the minimized cost -f
end