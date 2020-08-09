function [x, fun_val] = prob1_gradient_backtrack()


% Still trying to figure out what the format of the answer should be.
% My original approach is that answer should be a list of 101 3x1 vectors.
% The k=100 vector should be the minimum, and then we can graph them all.


% Run prob1_objective to load objective function f_obj and gradient grad_f.
prob1_objective;

% Set paramaters for backtracking iterator.
x = [1 1 1];
s = .5;
alpha = .1;
beta = 0.3;
epsilon = 1e-4;

% Run the backtracking iterator.
[x, fun_val] = backtrack_iterator(f_obj, grad_f, x, s, alpha, beta, epsilon);

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