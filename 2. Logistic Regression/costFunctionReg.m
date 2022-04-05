function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));


h = sigmoid(X*theta);

unreg_cost = ((-y)'*log(h) - (1-y)'*log(1-h))/m;

theta(1) = 0;

reg_cost = (lambda / (2 * m)) * (theta'*theta);

J = unreg_cost + reg_cost;

grad = (X'*(h - y) + lambda*theta)/m;


end
