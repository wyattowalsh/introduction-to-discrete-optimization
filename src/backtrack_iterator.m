function [x, fun_val] = backtrack_iterator(f, g, x0, s, alpha, beta, epsilon)
% Gradient method with backtracking stepsize rule
%
% INPUT
% f ......... objective function
% g ......... gradient of the objective function
% x0......... initial point
% s ......... initial choice of stepsize
% alpha ..... tolerance parameter for the stepsize selection
% beta ...... the constant in which the stepsize is multiplied at each backtracking step (0<beta<1)
% epsilon ... tolerance parameter for stopping rule
% ========================================================================
% OUTPUT
% x ......... optimal solution (up to a tolerance) of min f(x)
% fun_val ... optimal function value

% See page 57 for this backtracking setup in MATLAB (example 4.9).

x=x0; grad=g(x); fun_val=f(x); iter=0;
while (norm(grad)>epsilon) % stopping rule tolerance parameter satisifed
    iter=iter+1; % increase iterator count
    t=s; % set initial step size
    while (fun_val-f(x-t*grad)<alpha*t*norm(grad)^2) % test grad descent
        t=beta*t; % backtrack with coefficient beta
    end
    x=x-t*grad; % update x
    fun_val=f(x); % update fun_val
    grad=g(x); % update gradient
    fprintf('iter_number = %3d norm_grad = %2.6f fun_val = %2.6f \n',...
    iter,norm(grad),fun_val);
end