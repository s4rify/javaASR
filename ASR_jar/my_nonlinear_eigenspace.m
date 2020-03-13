function [Xsol, S0] = my_nonlinear_eigenspace(channels, X,  L, k, alpha)
% Example of nonlinear eigenvalue problem: total energy minimization.
%
% function Xsol = nonlinear_eigenspace(L, k, alpha)
%
% L is a discrete Laplacian operator,
% alpha is a given constant, and
% k corresponds to the dimension of the least eigenspace sought. 
%
% This example demonstrates how to use the Grassmann geometry factory 
% to solve the nonlinear eigenvalue problem as the optimization problem:
%
% minimize 0.5*trace(X'*L*X) + (alpha/4)*(rho(X)*L\(rho(X))) 
% over X such that X'*X = Identity,
%
% where L is of size n-by-n,
% X is an n-by-k matrix, and
% rho(X) is the diagonal part of X*X'.
%
% This example is motivated in the paper
% "A Riemannian Newton Algorithm for Nonlinear Eigenvalue Problems",
% Zhi Zhao, Zheng-Jian Bai, and Xiao-Qing Jin,
% SIAM Journal on Matrix Analysis and Applications, 36(2), 752-774, 2015.
%


% This file is part of Manopt and is copyrighted. See the license file.
%
% Main author: Bamdev Mishra, June 19, 2015.
% Contributors:
%
% Change log:

%
% my_nonlinear_eigenspace.m--
%
% Inputs: 
%
% Outputs:,   
%
% Example usage:,   
%
%
% Required files:,   

% Developed in Matlab 9.2.0.538062 (R2017a) on GLNXA64.
% Sarah Blum (sarah.blum@uni-oldenburg.de), 2017-12-14 15:13
%-------------------------------------------------------------------------

    % If no inputs are provided, generate a  discrete Laplacian operator.
    % This is for illustration purposes only.
    % The default example corresponds to Case (c) of Example 6.2 of the
    % above referenced paper.
    
    % I changed n, k = channels (Sarah Blum, 13.12.2017)
    if ~exist('L', 'var') || isempty(L)
        n = channels;
        L = gallery('tridiag', n, -1, 2, -1);
    end
    
    n = size(L, 1);
    assert(size(L, 2) == n, 'L must be square.');
    
    if ~exist('k', 'var') || isempty(k) || k > n
        k = channels;                                         
    end
    
    if ~exist('alpha', 'var') || isempty(alpha)
        alpha = 1;
    end
    

    
    % Grassmann manifold description
    Gr = grassmannfactory(n, k);
    problem.M = Gr;
    
       
    problem.cost  = @(Y)    -trace(Y'*X*Y);
    problem.egrad = @(Y)    -2*(X*Y); % Only Euclidean gradient needed.
    problem.ehess = @(Y, H) -2*(X*H); % Only Euclidean Hessian needed.
    
    p = 3; % p is some constant smaller n
    options.Delta_bar = 8*sqrt(p);
    options.tolgradnorm = 1e-7;
    options.verbosity = 0; % set to 0 to silence the solver, 2 for normal output.
    [Xsol, costXsol, info] = trustregions(problem, [], options); %#ok<ASGLU>
    [Vsol, Dsol] = eig(Xsol'*(X*Xsol));
    S0 = diag(Dsol);
    Xsol = Xsol*Vsol;
    
end
