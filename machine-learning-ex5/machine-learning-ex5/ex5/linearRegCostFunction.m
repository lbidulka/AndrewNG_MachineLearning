function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Cost
h_x = X*theta;
err = h_x - y;

J_reg = (lambda / (2*m)) * (sum(theta(2:end).^2));

J = (1 / (2*m)) * sum((h_x - y).^2) + J_reg;

% Gradient
for j = 1:columns(X)
    % Gradient regularization
    grad(j) = sum(err.*X(:,j)) + lambda*theta(j);
end

% Dont regularize 0 term
grad(1) = sum(err.*X(:,1));

grad = grad / m;

% =========================================================================

grad = grad(:);

end
