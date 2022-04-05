function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));


htheta = sigmoid(X * theta);
J = 1/m * sum(-y .* log(htheta) - (1 - y) .* log(1 - htheta));
% calculate grads
grad = (X'*(htheta - y))/m;

end
