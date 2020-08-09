% Uses symbolic tool box to model the objective function and its gradient.
% So far on my machine, I can only run the model for up to n = 100.
% Function also seems to output complex numbers, but I think it's right...

% Set dimension for objective function.
n = 100;

% Set symbolic mode for X as x1, x2, ... xn.
X = sym('x', [1, n]);

% Build array of variables to convert symbolic x1, x2, ... xn to a vector.
x_vars_array = [];
for i = 1:n
    x_vars_array = [x_vars_array; X(i)];
end

% Initialize symbolic function.
f_syms = 0;

% Construct function through iterative sum.
for i = 1:n
    summand_1 = log(5 - X(i)^2);
    random_a = rand(1, n);
    summand_2 = log(1 - 3 * (random_a * x_vars_array));
    f_syms = f_syms - summand_1 - summand_2;
end

% Initialize gradient as an array.
gradient_syms = [];

% Construct gradient iteratively by differentiating wrt X(i)
for i = 1:n
    gradient_syms = [gradient_syms; diff(f_syms, X(i))];
end

% Convert symbolic functions (f and grad) to matlab functions.
f = matlabFunction(f_syms, 'Vars', {x_vars_array});
g = matlabFunction(gradient_syms, 'Vars', {x_vars_array});

clear f_syms gradient_syms i n random_a summand_1 summand_2 X x_vars_array