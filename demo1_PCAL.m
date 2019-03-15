function demo1_PCAL
%% This demo shows how to call PCAL to solve
%       min  f(X), s.t.  X'MX=I.
%  where M is a symmetric positive definite matrix.
% -----------------------------------------
%   Demo 1:  M~=I (Identity)  (Generalized eigvalue problem)
%     ------------------------------------------------------------
%      f(X):= 0.5*trace(X'AX)
%     In fact, this problem is equivalent to find the p smallest generalized 
%     eigenvalues of matrix pencil (A,M), namely,
%           A*V = M*V*D.
%      We call MATLAB function 'eig' to verify our numerical results.
%     ------------------------------------------------------------
%% objective function
    function [F, G] = fun(X,  A)
        G = A*X;
        F = 0.5*sum(sum(X.*G));
    end
%% Problem generation
n = 200; p = 10;
A = randn(n); A = 0.5*(A+A');
R = randn(n); [Q,~] = qr(R);
M = Q'*diag(1.0001.^(1:n))*Q; M = 0.5*(M+M');

% parameters
opts.M = M;
opts.gtol = 1.e-10;
opts.info = 1;
% ----------------------------------
% The solver is sensitive to this penalty parameter.
% Users need to tune different values with different problem settings.
opts.penalparam = min(max(0.01,0.001*norm(A,2)),1e5);


% call solver
[X0,~] = qr(randn(n,p),0);
tic;[X,Out] = PCAL(X0,@fun,opts,A);t=toc;
fprintf('PCAL: CPU(s): %.5f, fval: %.10f, kkt: %3.2e, rerr: %3.2e,  feaX: %3.2e, iter: %d\n', ...
    t,Out.fval, Out.kkt, Out.xerr, Out.feaX, Out.iter);

% call 'eig' as baseline
tic;[U,~] = eig(A,M,'chol');t_eig=toc; 
V = U(:,1:p);[fval_eig,~] = fun(V,A);
fprintf('eigs: CPU(s): %.5f, fval: %.10f\n', ...
    t_eig,fval_eig);

fprintf('Subspace distance to exact solution: %3.2e\n',norm(X*X'-V*V','fro'))
end

