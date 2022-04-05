function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));




y_matrix = eye(num_labels)(y,:);

bias = ones(m, 1);
a1 = [bias, X]; z2 = a1 * Theta1';
a2 = [bias, sigmoid(z2)]; z3 = a2 * Theta2';
a3 = sigmoid(z3); h = a3;

J = (-1/m)*sum(sum((y_matrix.*log(h)) + ((1-y_matrix).*log(1-h))));

% Regularize Cost Function
reg = (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2)) + sum(sum((Theta2(:,2:end).^2))));

J = J + reg;



d3 = a3 - y_matrix;
g2 = sigmoidGradient(z2);

d2 = d3*Theta2(:,2:end).*g2; % (m x h)

del1 = d2'*a1;
del2 = d3'*a2;

Theta1_grad = (1/m)*del1;
Theta2_grad = (1/m)*del2;

% Regularized Gradient 

greg1 = Theta1*lambda/m;
greg2 = Theta2*lambda/m;

% Set first columns to zero (to ignore the biasing term)
greg1(:,1) = 0;
greg2(:,1) = 0;

Theta1_grad = Theta1_grad + greg1;
Theta2_grad = Theta2_grad + greg2;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
