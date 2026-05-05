% normal_procrustes_history.m
function [A, U, fval, history] = normal_procrustes_history(X, Y)
% Solves min_{A normal} ||AX - Y||_F^2 and returns convergence history.
%
% Inputs:
%   X, Y  -- m x n complex matrices
%
% Outputs:
%   A       -- optimal normal matrix (recovered as U D U*)
%   U       -- optimal unitary matrix
%   fval    -- final value of f(U)
%   history -- struct with fields:
%              .time     : CPU time at each iteration, shifted to start at 0
%              .residual : ||AX-Y||_F^2 at each iteration
%              .gradnorm : Riemannian gradient norm at each iteration

m = size(X, 1);

% Storage for history
time_list     = [];
cost_list     = [];
gradnorm_list = [];
t0 = tic;

% Define the manifold U(m)
M = unitaryfactory(m);

% Build the Manopt problem structure
problem.M = M;
problem.cost  = @(Uvar)       -f(Uvar, X, Y);
problem.egrad = @(Uvar)       -euclidean_grad(Uvar, X, Y);
problem.ehess = @(Uvar, Wvar) -euclidean_hess(Uvar, Wvar, X, Y, M);

% statsfun records time, cost and gradnorm at each iteration
options.statsfun  = @(problem, x, stats, store) record_stats(stats);
options.verbosity = 0;
options.Delta_bar = 1000;
options.maxiter   = 2000;
warning('off', 'manopt:getHessian:approx');

x0 = M.rand();
[U, neg_fval, ~, ~] = trustregions(problem, x0, options);
fval = -neg_fval;

% Extract U matrix if Manopt returns a struct
if isstruct(U), U = U.U; end

% Shift time so all trajectories start at t=0
history.time     = time_list - time_list(1);
history.residual = norm(Y,'fro')^2 + cost_list;
history.gradnorm = gradnorm_list;

% Recover optimal D using Lemma 1: d_i = (U*X)_i(U*Y)_i* / ||(U*X)_i||^2
UX = U' * X;
UY = U' * Y;
d = zeros(m, 1);
for i = 1:m
    phi_i = norm(UX(i,:))^2;
    if phi_i > 0
        d(i) = UX(i,:) * UY(i,:)' / phi_i;
    end
end
D = diag(d);
A = U * D * U';

    % Nested function to record stats at each iteration
    function stats = record_stats(stats)
        time_list(end+1, 1)     = toc(t0);
        cost_list(end+1, 1)     = stats.cost;
        gradnorm_list(end+1, 1) = stats.gradnorm;
    end
end

function val = f(U, X, Y)
% Objective: f(U) = sum_i |(U*Y)_i(U*X)_i*|^2 / ||(U*X)_i||^2
    if isstruct(U), U = U.U; end
    UX = U' * X;
    UY = U' * Y;
    val = 0;
    for i = 1:size(UX, 1)
        phi_i = norm(UX(i,:))^2;
        if phi_i > 0
            val = val + abs(UY(i,:) * UX(i,:)')^2 / phi_i;
        end
    end
end

function G = euclidean_grad(U, X, Y)
% Euclidean gradient of f at U
    if isstruct(U), U = U.U; end
    UX = U' * X;
    UY = U' * Y;
    m = size(UX, 1);
    phi = zeros(m, 1);
    lambda = zeros(m, 1);
    for i = 1:m
        phi(i) = norm(UX(i,:))^2;
        if phi(i) > 0
            lambda(i) = UY(i,:) * UX(i,:)';
        end
    end
    phi_inv = zeros(m, 1);
    phi_inv2 = zeros(m, 1);
    for i = 1:m
        if phi(i) > 0
            phi_inv(i) = 1 / phi(i);
            phi_inv2(i) = 1 / phi(i)^2;
        end
    end
    Phi_inv     = diag(phi_inv);
    Phi_inv2    = diag(phi_inv2);
    Lambda      = diag(lambda);
    Lambda_conj = diag(conj(lambda));
    Lambda_abs2 = diag(abs(lambda).^2);
    G = 2 * (X*Y'*U*Lambda*Phi_inv + Y*X'*U*Lambda_conj*Phi_inv - X*X'*U*Lambda_abs2*Phi_inv2);
end

function H = euclidean_hess(U, W, X, Y, M)
% Euclidean Hessian-vector product of f at U in direction W
    if isstruct(U), U = U.U; end
    W = M.tangent2ambient(U, W);
    UX = U' * X;  UY = U' * Y;
    WX = W' * X;  WY = W' * Y;
    m = size(UX, 1);
    phi = zeros(m, 1);
    lambda = zeros(m, 1);
    for i = 1:m
        phi(i) = norm(UX(i,:))^2;
        if phi(i) > 0
            lambda(i) = UY(i,:) * UX(i,:)';
        end
    end
    phi_inv = zeros(m, 1);
    phi_inv2 = zeros(m, 1);
    for i = 1:m
        if phi(i) > 0
            phi_inv(i) = 1 / phi(i);
            phi_inv2(i) = 1 / phi(i)^2;
        end
    end
    Phi_inv     = diag(phi_inv);
    Phi_inv2    = diag(phi_inv2);
    Lambda      = diag(lambda);
    Lambda_conj = diag(conj(lambda));
    Lambda_abs2 = diag(abs(lambda).^2);
    delta_lambda = zeros(m, 1);
    delta_phi    = zeros(m, 1);
    for i = 1:m
        if phi(i) > 0
            delta_lambda(i) = WY(i,:)*UX(i,:)' + UY(i,:)*WX(i,:)';
            delta_phi(i)    = 2*real(UX(i,:)*WX(i,:)');
        end
    end
    dLambda      = diag(delta_lambda);
    dLambda_conj = diag(conj(delta_lambda));
    dPhi         = diag(delta_phi);
    dLambda_abs2 = dLambda*Lambda_conj + Lambda*dLambda_conj;
    dPhi_inv     = -Phi_inv  * dPhi * Phi_inv;
    dPhi_inv2    = -Phi_inv  * dPhi * Phi_inv2 - Phi_inv2 * dPhi * Phi_inv;
    H = 2 * ( ...
        X*Y'*W*Lambda*Phi_inv      + X*Y'*U*dLambda*Phi_inv      + X*Y'*U*Lambda*dPhi_inv      + ...
        Y*X'*W*Lambda_conj*Phi_inv + Y*X'*U*dLambda_conj*Phi_inv + Y*X'*U*Lambda_conj*dPhi_inv - ...
        X*X'*W*Lambda_abs2*Phi_inv2 - X*X'*U*dLambda_abs2*Phi_inv2 - X*X'*U*Lambda_abs2*dPhi_inv2 ...
    );
end

function S = skew(X)
    S = (X - X') / 2;
end