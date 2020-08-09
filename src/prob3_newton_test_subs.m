%% Calculating objective function, its gradient, and its Hessian.
% This method uses pre-allocated symbolic matrices, which should be faster.

% Set dimension n for the objective function.
n = 10;

% Initialize X as a vector of symbolic xi values.
X = sym('x', [n, 1]);

% Initialize symbolic function.
f_syms = 0;
% Construct the objective function iteratively.
for i = 1:n
    summand_1 = log(5 - X(i)^2);
    random_a = rand(1, n);
    summand_2 = log(1 - 3 * (random_a * X));
    f_syms = f_syms  - summand_1 - summand_2;
end

grad = sym(zeros(n,1)); % pre-allocate size of gradient symbolic vector.
% Construct the gradient iteratively.
for j = 1:n
    grad(j) = diff(f_syms, X(j));
end

hess = sym(zeros(n,n)); % pre-allocate size of Hessian symbolic matrix.
% Construct the gradient iteratively.
for i = 1:n
    for j =1:n
        hess(i,j) = diff(grad(i),X(j));
    end
end

clear i j random_a summand_1 summand_2

%% Newton method with backtracking line search.

% Initialize backtracking parameters.
x0 = rand(n,1);
s = 0.5;
iter = 0;
alpha = 0.1;
beta = 0.3;
epsilon = 1e-4;
x = x0;

% Initialize Newton values.
value = double(subs(f_syms, X, x));
grad = double(subs(grad, X, x));
hess = double(subs(hess, X, x));

% Initialize step-recording vector.
xx = zeros(1,300);

d = hess\grad;
while (norm(grad)>epsilon) 
    iter=iter+1;
    t=s;
    nv = x-t*d;
    update_f = double(subs(f_syms, X, nv));
    while update_f > value-alpha*t*grad'*d
        t = beta*t;
        nv = x-t*d;
        update_f = double(subs(f_syms, X, nv));
    end
    x = x-t*d;
    d = hess\grad;
    value = double(subs(f_syms,X,x));
    grad = double(subs(grad,X,x));
    
    fprintf('Iterations = %d, Norm of grad = %d, FN Value = %d \n', iter,norm(grad),value)
    xx(iter) = double(subs(f_syms, X, x));
    
end
k = 1:300;
figure(1)
semilogx(k,abs(xx))
%}