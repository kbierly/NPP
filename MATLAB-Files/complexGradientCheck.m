% Numerically verify the Euclidean gradient formula for the complex
% Normal Procrustes Problem (NPP) using Manopt's checkgradient.
%
% Objective (to maximize):
%   f(U) = sum_i |gamma_i|^2 / phi_i,
%   gamma_i = (U^*Y)_i (U^*X)_i^*,  phi_i = ||(U^*X)_i||^2,
% with optimal diagonal d_i^* = gamma_i / phi_i.
%
% Manopt *minimizes*, so we pass cost = -f and egrad = -grad f.
%
% Runs the check on three problem sizes: (m,n) = (50,50), (50,25), (50,10).
% checkgradient draws the first-order Taylor error E(t) on a log-log plot
% and reports the slope, and we display the slope in the figure title. 
%
% Requires Manopt on the path (https://www.manopt.org).
function complexGradientCheck()
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
        
        pngFileName = sprintf('GradientCheck_m%d_n%d.png', m, n);
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