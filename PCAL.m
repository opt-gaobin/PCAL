function [X,Out] = PCAL(X, fun, opts, varargin)
% -----------------------------------------------------------------------
% Using a Parallel Column-wise Block Minimization
% (for Proximal Linearized Augmented Lagrangian )
% to solve
%
%       min  f(X),  s. t.  X'MX = I,  where X\in R^{n,p}
%
%  where M is a symmetric positive definite matrix.
% ----------------------------------
%  Input:
%                         X --- n-by-p initial matrix such that X'*M*X = I
%                     fun --- a matlab function for f(X)
%                                    call: [funX,F] = fun(X,data1,data2);
%                                    funX: function value f(X)
%                                           F:  gradient of f(X)
%                                    data: extra data (can be more)
%          varargin --- data1, data2

%
%  Calling syntax:
%      If M is an identity matrix, i.e., X'X = I.
%       opts.M=[]; otherwise, opts.M=M;
%      [X, out]= PCAL(X0, @fun, opts, data1, data2);    
%
%                     opts --- option structure with fields:
%                                               M:  n-by-n symmetric positive definite matrix
%                                        xtol:  stop control for ||X_k - X_{k+1}||/sqrt(n)
%                                        gtol:  stop control for ||kkt||/||kkt0||
%                                        ftol:  stop control for |f_k - f_{k+1}|/(|f_k|+1)
%                               stepsize:  0(ABB stepsize) o.w.(fixed stepsize)
%                           penalparam:  penalty factor
%                                                     This solver is sensitive to penalty parameter. Users need
%                                                      to tune different values with different problem settings.
%                                    solver:  1 (PCAL)  2 (PLAM)   3 (PCAL-S, simplified multiplier)
%                                postorth:  1 (post-procedure)  0 (no post-procedure)
%                                      maxit:  max iteration
%                                        info:   0(no print) o.w.(print)
%
%  Output:
%                            X --- solution
%                        Out --- output information
%                                        kkt: ||kkt|| (first-order optimality condition)
%                                      fval:  function value of solution
%                                      feaX: ||I-X'X||_F (feasiblity violation)
%                                      xerr: ||X_k - X_{k+1}||/sqr(n)
%                                      iter: total iteration number
%                                    fvals: history of function value
%                                    feaXs: history of feasibility
%                                      kkts: history of kkt
%                                message: convergence message
% --------------------------------------------------------------------
%  Reference:
%  B. Gao, X. Liu, and Y.-x. Yuan, Parallelizable algorithms for optimization
%  problems with orthogonality constraints, arXiv preprint arXiv:1810.03930, (2018).
% ----------------------------------
%  Author: Bin Gao, Xin Liu (ICMSEC, AMSS, CAS)
%                 gaobin@lsec.cc.ac.cn
%                 liuxin@lsec.cc.ac.cn
%
%  Version: 1.0 --- 2016/12/22
%  Version: 1.1 --- 2018/02/20: support general function
%  Version: 1.2 --- 2019/01/29: support X'MX=I
%---------------------------------------------------------------
%% default setting
if nargin < 3;opts=[];end

if isempty(X)
    error('input X is an empty matrix');
else
    [n, p] = size(X);
end

if isfield(opts, 'M') && ~isempty(opts.M)
    if ~issymmetric(opts.M);error('input M is not a symmetric matrix');end
    general_flag = 1;
else
    general_flag = 0;
end

if isfield(opts, 'xtol')
    if opts.xtol < 0 || opts.xtol > 1
        opts.xtol = 1e-10;
    end
else
    opts.xtol = 1e-10;
end

if isfield(opts, 'gtol')
    if opts.gtol < 0 || opts.gtol > 1
        opts.gtol = 1e-6;
    end
else
    opts.gtol = 1e-6;
end

if isfield(opts, 'ftol')
    if opts.ftol < 0 || opts.ftol > 1
        opts.ftol = 1e-12;
    end
else
    opts.ftol = 1e-12;
end

if isfield(opts, 'proxparam')
    if opts.proxparam < 0
        opts.proxparam = 0;
    end
else
    opts.proxparam = 0;
end

if isfield(opts, 'stepsize')
    if opts.stepsize < 0
        opts.stepsize = 0;
    end
else
    opts.stepsize = 0;
end

% 1.PCAL  2.PLAM  3.PCAL-S
if isfield(opts, 'solver')
    if all(opts.solver ~= 1:2)
        opts.solver = 1;
    end
else
    opts.solver = 1;
end

if isfield(opts, 'postorth')
    if all(opts.postorth ~= [0 1])
        opts.postorth = 1;
    end
else
    opts.postorth = 1;
end

if isfield(opts, 'maxit')
    if opts.maxit < 0 || opts.maxit > 1.e10
        opts.maxit = 1000;
    end
else
    opts.maxit = 1000;
end

if ~isfield(opts, 'info');opts.info = 0;end

%% ---------------------------------------------------------------
% copy parameters
xtol = opts.xtol;
gtol = opts.gtol;
ftol = opts.ftol;
stepsize = opts.stepsize;
penalparam = opts.penalparam;
solver = opts.solver;
postorth = opts.postorth;
maxit = opts.maxit;
info = opts.info;

global Ip
Ip = speye(p);

%% ---------------------------------------------------------------
% Initialization
iter = 0; iter_final=0;
Out.fvals = []; Out.kkts = []; Out.feaXs = []; Out.Xs = [];
% evaluate function and gradient info.
[funX,G] = feval(fun, X , varargin{:});
if general_flag; M = opts.M; MX = M*X; else; MX = X;end
[PL,kktval0,feaX] = getPG(X,G,MX);
% save history
Out.fvals(1) = funX; Out.kkts(1) = kktval0;
Out.feaXs(1) = feaX; Out.Xs{1} = X;

% initial stepsize
if stepsize == 0
    proxparam = 1/max(0.1,min(0.01*norm(PL,'fro'),1));
else
    proxparam = 1/stepsize;
end

% info
if info ~= 0
    switch solver
        case 1
            fprintf('------------------ PCAL start ------------------\n');
        case 2
            fprintf('------------------ PLAM start ------------------\n');
        case 3
            fprintf('------------------ PCAL-S start ------------------\n');
    end
    fprintf('%4s | %15s | %10s | %10s | %8s | %8s\n', 'Iter ', 'F(X) ', 'KKT ', 'Xerr ', 'Feasi ', 'tau');
    fprintf('%d \t %f \t %3.2e \t %3.2e \t %3.2e \t %3.2e\n',iter, funX, kktval0, 0, feaX, 1/proxparam);
end

%% main iteration
for iter = 1: maxit
    Xk = X; PLk = PL;   funXk = funX;
    
    % one gradient step for PCAL, PLAM
    X = Xk - PLk/proxparam;
    if general_flag
        MX = M*X;
    else
        MX = X;
    end
    
    % ------------ Only for PCAL ------------
    if any(solver == [1 3])
        % -------- solve subproblems -----------
        v = sqrt(sum(X.*MX)); % column square root
        X = X./v;
        if general_flag
            MX = MX./v;
        else
            MX = X;
        end
    end
    
    % ------------ evaluate error ------------
    [funX,G] = feval(fun, X , varargin{:});
    [PL,kktval,feaX] = getPG(X,G,MX);
    
    % ------------ save history ------------
    Out.fvals(iter+1) = funX;
    Out.feaXs(iter+1) = feaX;
    Out.kkts(iter+1) = kktval;
    xerr = norm(Xk - X,'fro')/sqrt(n);
    ferr = abs(funXk - funX)/(abs(funXk)+1);  
    % info
    if info ~= 0 && (mod(iter,15) == 0 )
        fprintf('%d \t %f \t %3.2e \t %3.2e \t %3.2e \t %3.2e\n',iter, funX, kktval, xerr, feaX, 1/proxparam);
    end
    
    % --------- parameter: proxparam -------------
    % BB1(-1)  ABB(-2) Difference(-3)  BB2(-4) constant (any scalar>0)
    if stepsize == 0
        Sk = X-Xk;
        Vk = PL-PLk;    % Vk = G-Gk;
        SV = sum(sum(Sk.*Vk));
        if mod(iter+1,2) == 0
            proxparam = sum(sum(Vk.*Vk))/abs(SV); % SBB for odd
        else
            proxparam = abs(SV)/sum(sum(Sk.*Sk)); % LBB for even
        end
        proxparam = max(1.e-10,min(proxparam,1.e10));
    end
    
    % ------------------ stop criteria --------------------
    %     if (kktval/kktval0 < gtol)
    %     if (kktval/kktval0 < gtol && feaX < 1.e-12)
    if (kktval/kktval0 < gtol)  || (xerr < xtol && ferr < ftol)
        Out.message = 'converge';
        iter_final = iter;
        % ----- post-procedure: orthogonal step -----
        if postorth == 1
            if feaX > 1e-13
                % projection to orthogonality constraints
                if general_flag
                    R = chol(M); RX = R*X;
                    [U,~,V] = svd(RX,0); B = U*V';  % projection
                    opts_orth.UT = true; X = linsolve(R,B,opts_orth); % solve RX = B
                else
                    [U,~,V] = svd(X,0);   X = U*V';   % projection
                end
                
                iter_final = iter+1;
                [funX,G] = feval(fun, X , varargin{:});
                if general_flag; MX = M*X; else; MX = X;end
                [~,kktval,feaX] = getPG(X,G,MX);
                Out.fvals(iter_final) = funX;
                Out.feaXs(iter_final) = feaX;
                Out.kkts(iter_final) = kktval;
                xerr = norm(Xk - X,'fro')/sqrt(n);
                ferr = abs(funXk - funX)/(abs(funXk)+1);
            end
        end
        break;
    end
    
    if iter >= maxit
        iter_final = maxit;
        Out.message = 'exceed max iteration';
        break;
    end
    
    % ------------- adaptive parameter: penalparam ------------
    %     if  feaX > 1.e-13
    %         penalparam = 2*penalparam;
    %     end
    % ------------------------------------------
    
end

Out.feaX = feaX;
Out.fval = funX;
Out.iter = iter_final;
Out.xerr = xerr;
Out.kkt  = kktval;
Out.ferr = ferr;

if info ~= 0
    if iter_final > iter
        fprintf('%s at %d-th and post-procedure is calling...\n',Out.message,iter_final-1);
    else
        fprintf('%s at ... (without post-procedure)\n',Out.message);
    end
    fprintf('%d \t %f \t %3.2e \t %3.2e \t %3.2e \t %3.2e\n',iter_final, funX, kktval, xerr, feaX, 1/proxparam);
    fprintf('------------------------------------------------------------------------\n');
end

%% ---------------------------------------------------------------
% nest-function
% get Lagrangian gradient: PL
% ||G-MXG'X||: kktval
% ||I-X'MX||: feaX
    function [PL,kktval,feaX] = getPG(X,G,MX)
        GX = G'*X;  % grad'*X
        GXsym = 0.5*(GX+GX');
        XGXsym = MX*GXsym;
        XX = X'*MX;
        FeaX = XX-Ip;  % X'X-I
        
        kkt = G - XGXsym;
        kktval = norm(kkt,'fro');
        feaX = norm(FeaX,'fro');
        
        if penalparam ~= 0
            penalFeaX = penalparam*FeaX; % beta*(X'MX-I)
        end
        
        % -------- Lambda & PL --------
        switch solver
            case 1   % PCAL
                %  PG = G - MX*GXsym; % kkt  (G-MXG'X)
                %  Lambda = GXsym + diag(diag(MX'*(PG-penalPX)));
                if penalparam ~= 0
                    d = diag(GX'-XX*GXsym+XX*penalFeaX);
                    % Lambda = GXsym + diag(d);
                    PL = kkt - MX.*d' + MX*penalFeaX;                    
                else
                    d = diag(GX'-XX*GXsym);
                    % Lambda = GXsym + diag(d);
                    PL = kkt - MX.*d'; % Lambda = XGXsym + XD;
                end     
            otherwise  % PLAM, PCAL-S
                % Lambda = GXsym;
                if penalparam ~= 0
                    PL = kkt + MX*penalFeaX;
                else
                    PL = kkt;
                end
        end
        % -------------------
    end
% -------------------------------------------------------
end