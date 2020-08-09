function [x, value, xx] = NLDOgradient1()
f = @(x1,x2,x3) (exp(x1+1) + exp(-2*x1+1) + exp(x2+1) + exp(-2*x2+1) + exp(x3+1) ...
    + exp(-2*x3+1) + (x1 + 4*x2 + 6*x3)^4);
gradf = @(x1,x2,x3) [(exp(x1+1) + (-2)* exp(-2*x1+1) + 4*(x1 + 4*x2 + 6*x3)^3);...
    (exp(x2+1) +(-2)*exp(-2*x2+1) + 16*(x1 + 4*x2 + 6*x3)^3);...
    (exp(x3+1) +(-2)*exp(-2*x3+1) +24*(x1 + 4*x2 + 6*x3)^3)];
x0 = zeros(3,1);
s = 0.5;
iter = 0;
alpha = 0.1;
beta = 0.3;
epsilon = 1e-4;
x = x0;
x1 = x(1);
x2 = x(2);
x3 = x(3);
value = f(x1,x2,x3);
grad = gradf(x1,x2,x3);
xx = zeros(1,100);
points = zeros(3,100);
points(1:3,1) = x;
while (norm(grad)>epsilon) 
    iter=iter+1;
    t=s;
    nv = x-(t*grad);
    while (value - f(nv(1),nv(2),nv(3)) < alpha*t*norm(grad)^2)
        t = beta*t;
       nv = x-(t*grad);
    end
    x=x-t*grad;
    points(1:3,iter+1) = x;
    value = f(x(1),x(2),x(3));
    grad = gradf(x(1),x(2),x(3));
    fprintf('Iterations = %d, Norm of grad = %d, FN Value = %d \n', iter,norm(grad),value)
    xx(iter) = value; 
end
xx(iter:100) = xx(iter);
points(1,iter:100) = points(1,iter);
points(2,iter:100) = points(2,iter);
points(3,iter:100) = points(3,iter);
k = 1:100;
figure(1)
semilogx(k,xx)
figure(2)
plot3(points(1,:),points(2,:),points(3,:))
end

