function [x, value] = NLDOhw3_3()
%% Calculating the function, gradient, and Hessian

n = 500;
X = sym(zeros(n,1)); %sym() makes matrix symbolic
X = sym('x', [n, 1]);
f = 0;

for i = 1:n
    summand_1 = log(5 - X(i)^2);
    random_a = rand(1, n);
    summand_2 = log(1 - 3 * (random_a * X));
    f = f  - summand_1 - summand_2;
end

grad = sym(zeros(n,1)); %pre-allocate
for j = 1:n
    grad(j) = diff(f, X(j));
end

hess = sym(zeros(n,n)); %pre-allocate
for i = 1:n
    for j =1:n
        hess(i,j) = diff(grad(i),X(j));
    end
end
%% Using Newton's Method
%basically same as #2 with different size k and value matrices; no second figure

x0 = zeros(n,1);
s = 0.5;
iter = 0;
alpha = 0.1;
beta = 0.3;
epsilon = 1e-4;
x = x0;
value = subs(f,X,x);
grad = subs(grad,X,x);
hess = subs(hess,X,x);
xx = zeros(1,300);
d = hess\grad;
while (norm(grad)>epsilon) 
    iter=iter+1;
    t=s;
    nv = x-t*d;
    while f(nv(1),nv(2),nv(3))> value-alpha*t*grad'*d
        t = beta*t;
        nv = x-t*d;
    end
    x = x-t*d;
    d = hess\grad;
    value = subs(f,X,x);
    grad = subs(grad,X,x);
    fprintf('Iterations = %d, Norm of grad = %d, FN Value = %d \n', iter,norm(grad),value)
    xx(iter) = value; 
    
end
k = 1:300;
figure(1)
semilogx(k,abs(xx))
end
