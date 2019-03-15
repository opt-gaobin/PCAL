function demo2_PCAL
%% This demo shows how to call PCAL to solve
%       min  f(X), s.t.  X'MX=I.
%  where M is a symmetric positive definite matrix.
% -----------------------------------------
%   Demo 1:  M=I (Identity)  (nonlinear eigvalue problem)
%     ------------------------------------------------------------
%     f(X) = E(X) := 0.5*trace(X'AX)+0.25*alpha*rho(X)'inv(A)*rho(X)
%          where    rho(X) = diag(X*X')
%     ------------------------------------------------------------
%% objective function
    function [funX, F] = fun(X, A,invA, alpha)
        AX = A*X;
        rhoX = sum(X.*X,2);
        Ainvrho = invA*rhoX;
        Fh = diag(alpha*Ainvrho)*X;
        F = AX+Fh;
        funX = 0.5*sum(sum(X.*AX)) + 0.25*alpha*dot(rhoX,Ainvrho);
    end
%% Problem generation
n = 500;  p = 10;
A = randn(n); A = 0.5*(A+A'); invA = pinv(A);   alpha = 1;
opts = [];  opts.info = 1;
opts.gtol = 1e-5;

% parameters
opts.M = [];
opts.gtol = 1.e-10;
opts.info = 1;
% ----------------------------------
% The solver is sensitive to this penalty parameter.
% Users need to tune different values with different problem settings.
opts.penalparam = min(max(0.01,0.001*norm(A)),1e5);


% call solver
[X0,~] = qr(randn(n,p),0);
tic;[~,Out] = PCAL(X0,@fun,opts,A,invA, alpha);t=toc;
fprintf('PCAL: CPU(s): %.5f, fval: %.10f, kkt: %3.2e, rerr: %3.2e,  feaX: %3.2e, iter: %d\n', ...
    t,Out.fval, Out.kkt, Out.xerr, Out.feaX, Out.iter);
end

