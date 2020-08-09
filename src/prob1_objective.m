% Writes out the objective function f_obj calculates its gradient grad_f.

% Set symbolic mode for three variables.
X = sym('x', [1, 3]);

% Translate function to MATLAB syntax. Use diff() to calculate grad, hess.
f_syms = exp(X(1) + 1) + exp(-2*X(1) + 1) + exp(X(2) + 1) + exp(-2*X(2) + 1) + exp(X(3) + 1) + exp(-2*X(3) + 1) + (X(1) + 4*X(2) + 6*X(3))^4;
gradient_syms = [diff(f_syms, X(1)); diff(f_syms, X(2)); diff(f_syms, X(3))];

% Convert symbolic expressions of f, grad, hess to functions.
f_obj = matlabFunction(f_syms, 'Vars', {[X(1) X(2) X(3)]});
grad_f = matlabFunction(gradient_syms, 'Vars', {[X(1) X(2) X(3)]});

% Clear extraneous variables from workspace.
clear X f_syms gradient_syms;