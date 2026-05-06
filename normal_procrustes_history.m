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
    if isstruct(U), U = U.U; end
    UX = U' * X;
    UY = U' * Y;
    m  = size(UX, 1);
    d  = zeros(m, 1);
    for i = 1:m
        phi_i = norm(UX(i,:))^2;
        if phi_i > 0
            d(i) = UX(i,:) * UY(i,:)' / phi_i;  % d_i* = (U*X)_i(U*Y)_i* / phi_i
        end
    end
    Dstar      = diag(d);
    Dstar_conj = diag(conj(d));   % (D*)*
    Dstar_abs2 = diag(abs(d).^2); % |D*|^2
    % grad = 2( XY* U (D*)* + YX* U D* - XX* U |D*|^2 )
    G = 2 * (X*Y'*U*Dstar_conj + Y*X'*U*Dstar - X*X'*U*Dstar_abs2);
end

function H = euclidean_hess(U, W, X, Y, M)
    if isstruct(U), U = U.U; end
    W  = M.tangent2ambient(U, W);
    UX = U' * X;  UY = U' * Y;
    WX = W' * X;  WY = W' * Y;
    m  = size(UX, 1);

    phi    = zeros(m, 1);
    lambda = zeros(m, 1);
    d      = zeros(m, 1);
    for i = 1:m
        phi(i) = norm(UX(i,:))^2;
        if phi(i) > 0
            lambda(i) = UY(i,:) * UX(i,:)';       % Lambda_ii = (U*Y)_i(U*X)_i*
            d(i)      = conj(lambda(i)) / phi(i);  % d_i* = Lambda_ii* / phi_i
        end
    end

    delta_lambda = zeros(m, 1);
    delta_phi    = zeros(m, 1);
    for i = 1:m
        if phi(i) > 0
            delta_lambda(i) = WY(i,:)*UX(i,:)' + UY(i,:)*WX(i,:)';
            delta_phi(i)    = 2*real(UX(i,:)*WX(i,:)');
        end
    end

    % delta(d_i*) = conj(delta_lambda_i)/phi_i - conj(lambda_i)/phi_i^2 * delta_phi_i
    delta_d = zeros(m, 1);
    for i = 1:m
        if phi(i) > 0
            delta_d(i) = conj(delta_lambda(i)) / phi(i) ...
                       - conj(lambda(i)) / phi(i)^2 * delta_phi(i);
        end
    end

    Dstar      = diag(d);
    Dstar_conj = diag(conj(d));
    Dstar_abs2 = diag(abs(d).^2);

    dDstar      = diag(delta_d);
    dDstar_conj = diag(conj(delta_d));
    % delta(|D*|^2) = delta(D*)(D*)* + D* delta(D*)*
    dDstar_abs2 = dDstar*Dstar_conj + Dstar*dDstar_conj;

    % Hess = 2( XY* W (D*)* + XY* U delta(D*)*
    %         + YX* W D*    + YX* U delta(D*)
    %         - XX* W |D*|^2 - XX* U delta(|D*|^2) )
    H = 2 * ( ...
        X*Y'*W*Dstar_conj  + X*Y'*U*dDstar_conj  + ...
        Y*X'*W*Dstar       + Y*X'*U*dDstar        - ...
        X*X'*W*Dstar_abs2  - X*X'*U*dDstar_abs2 );
end

function S = skew(X)
    S = (X - X') / 2;
end