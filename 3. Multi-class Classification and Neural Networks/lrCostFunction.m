function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


h_theta_x =  sigmoid(X * theta);

J = (1/m) * sum(- y.* log(h_theta_x) - (1 - y).* log(1 - h_theta_x));

J = J + (lambda / (2 * m)) * sum((theta(2:end,:)).^2);

grad = (1/m) * (X' * (h_theta_x - y));

grad(2:end,:) = grad(2:end,:) + (lambda / m) * theta(2:end,:);


grad = grad(:);

end
