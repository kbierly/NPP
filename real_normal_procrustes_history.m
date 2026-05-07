function [A, Q, gval, history] = real_normal_procrustes_history(X, Y)
% real_normal_procrustes_history.m
% Solves min_{A real normal} ||AX - Y||_F^2 and returns convergence history.
%
% Every real normal matrix admits A = Q*Delta*Q^T (Theorem 2.5.8, Horn & Johnson)
% where Q in O(m) and Delta is block diagonal with 2x2 rotation-scaling blocks
%   [a_i, b_i; -b_i, a_i]  (eigenvalues a_i +/- b_i*i)
% plus one 1x1 block (lambda) if m is odd.
%
% For fixed Q, the optimal Delta is given in closed form by Lemma 4.1.
% We maximize g(Q) over O(m) via Manopt trustregions.
%
% Inputs:
%   X, Y  -- m x n real matrices
%
% Outputs:
%   A       -- optimal real normal matrix (Q*Delta*Q^T)
%   Q       -- optimal orthogonal matrix
%   gval    -- final value of g(Q)
%   history -- struct with fields .time, .residual, .gradnorm

m = size(X, 1);

time_list     = [];
cost_list     = [];
gradnorm_list = [];
t0 = tic;

% O(m) via Stiefel manifold St(m,m)
M = stiefelfactory(m, m);

problem.M     = M;
problem.cost  = @(Qvar)       -g(Qvar, X, Y);
problem.egrad = @(Qvar)       -euclidean_grad_real(Qvar, X, Y);
problem.ehess = @(Qvar, Wvar) -euclidean_hess_real(Qvar, Wvar, X, Y, M);

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

% Recover A = Q * Delta*(Q) * Q^T
Delta = build_Delta(Q, X, Y);
A     = Q * Delta * Q';

    function stats = record_stats(stats)
        time_list(end+1, 1)     = toc(t0);
        cost_list(end+1, 1)     = stats.cost;
        gradnorm_list(end+1, 1) = stats.gradnorm;
    end
end

% =========================================================================

function val = g(Q, X, Y)
% g(Q) = sum_i (alpha_i^2 + beta_i^2)/phi_i + (1x1 term if m odd)
% where alpha_i, beta_i, phi_i are from Proposition 4.2.
    QX  = Q' * X;
    QY  = Q' * Y;
    m   = size(QX, 1);
    val = 0;
    for i = 1:floor(m/2)
        phi_i = norm(QX(2*i-1,:))^2 + norm(QX(2*i,:))^2;
        if phi_i > 0
            alpha = QX(2*i-1,:)*QY(2*i-1,:)' + QX(2*i,:)*QY(2*i,:)';
            beta  = QX(2*i,:)*QY(2*i-1,:)'   - QX(2*i-1,:)*QY(2*i,:)';
            val   = val + (alpha^2 + beta^2) / phi_i;
        end
    end
    if mod(m, 2) == 1
        phi_m = norm(QX(m,:))^2;
        if phi_m > 0
            val = val + (QX(m,:)*QY(m,:)')^2 / phi_m;
        end
    end
end

% =========================================================================

function Delta = build_Delta(Q, X, Y)
% Build Delta*(Q) from Lemma 4.1.
    QX    = Q' * X;
    QY    = Q' * Y;
    m     = size(QX, 1);
    Delta = zeros(m, m);
    for i = 1:floor(m/2)
        phi_i = norm(QX(2*i-1,:))^2 + norm(QX(2*i,:))^2;
        if phi_i > 0
            a = (QX(2*i-1,:)*QY(2*i-1,:)' + QX(2*i,:)*QY(2*i,:)') / phi_i;
            b = (QX(2*i,:)*QY(2*i-1,:)'   - QX(2*i-1,:)*QY(2*i,:)') / phi_i;
            Delta(2*i-1:2*i, 2*i-1:2*i) = [a, b; -b, a];
        end
    end
    if mod(m, 2) == 1
        phi_m = norm(QX(m,:))^2;
        if phi_m > 0
            Delta(m,m) = QX(m,:)*QY(m,:)' / phi_m;
        end
    end
end

% =========================================================================

function G = euclidean_grad_real(Q, X, Y)
% Euclidean gradient (eq. real_euclidean_grad):
%   nabla_Q g = 2(XY^T Q Delta* + YX^T Q Delta*^T - XX^T Q Delta*^T Delta*)
    Delta       = build_Delta(Q, X, Y);
    DeltaT      = Delta';
    DeltaTDelta = DeltaT * Delta;
    G = 2 * (X*Y'*Q*Delta + Y*X'*Q*DeltaT - X*X'*Q*DeltaTDelta);
end

% =========================================================================

function H = euclidean_hess_real(Q, W, X, Y, M)
% Euclidean Hessian-vector product (eq. real_hess):
%   Hess_Q g[W] = 2( XY^T W Delta*          + XY^T Q delta(Delta*)
%                  + YX^T W Delta*^T         + YX^T Q delta(Delta*^T)
%                  - XX^T W Delta*^T Delta*  - XX^T Q delta(Delta*^T Delta*) )
    W   = M.tangent2ambient(Q, W);
    QX  = Q' * X;  QY = Q' * Y;
    WX  = W' * X;  WY = W' * Y;
    m   = size(QX, 1);

    Delta  = zeros(m, m);
    dDelta = zeros(m, m);

    for i = 1:floor(m/2)
        phi_i = norm(QX(2*i-1,:))^2 + norm(QX(2*i,:))^2;
        if phi_i > 0
            alpha = QX(2*i-1,:)*QY(2*i-1,:)' + QX(2*i,:)*QY(2*i,:)';
            beta  = QX(2*i,:)*QY(2*i-1,:)'   - QX(2*i-1,:)*QY(2*i,:)';
            a     = alpha / phi_i;
            b     = beta  / phi_i;
            Delta(2*i-1:2*i, 2*i-1:2*i) = [a, b; -b, a];

            % delta(alpha_i), delta(beta_i), delta(phi_i) with O=W
            d_alpha = WX(2*i-1,:)*QY(2*i-1,:)' + WX(2*i,:)*QY(2*i,:)' ...
                    + QX(2*i-1,:)*WY(2*i-1,:)' + QX(2*i,:)*WY(2*i,:)';
            d_beta  = WX(2*i,:)*QY(2*i-1,:)'   - WX(2*i-1,:)*QY(2*i,:)' ...
                    + QX(2*i,:)*WY(2*i-1,:)'   - QX(2*i-1,:)*WY(2*i,:)';
            d_phi   = 2*(QX(2*i-1,:)*WX(2*i-1,:)' + QX(2*i,:)*WX(2*i,:)');

            % delta(a_i*) = d_alpha/phi - a*d_phi/phi
            % delta(b_i*) = d_beta/phi  - b*d_phi/phi
            da = d_alpha/phi_i - a*d_phi/phi_i;
            db = d_beta /phi_i - b*d_phi/phi_i;
            dDelta(2*i-1:2*i, 2*i-1:2*i) = [da, db; -db, da];
        end
    end

    if mod(m, 2) == 1
        phi_m = norm(QX(m,:))^2;
        if phi_m > 0
            lam        = QX(m,:)*QY(m,:)' / phi_m;
            Delta(m,m) = lam;
            d_alpha_m  = WX(m,:)*QY(m,:)' + QX(m,:)*WY(m,:)';
            d_phi_m    = 2*(QX(m,:)*WX(m,:)');
            dDelta(m,m) = d_alpha_m/phi_m - lam*d_phi_m/phi_m;
        end
    end

    DeltaT       = Delta';
    dDeltaT      = dDelta';
    % delta(Delta*^T Delta*) = delta(Delta*^T)*Delta* + Delta*^T*delta(Delta*)
    dDeltaTDelta = dDeltaT*Delta + DeltaT*dDelta;

    H = 2 * ( X*Y'*W*Delta            + X*Y'*Q*dDelta          + ...
              Y*X'*W*DeltaT           + Y*X'*Q*dDeltaT         - ...
              X*X'*W*(DeltaT*Delta)   - X*X'*Q*dDeltaTDelta );
end