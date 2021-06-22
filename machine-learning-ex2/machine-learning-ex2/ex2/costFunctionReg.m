function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% Cost
h_x = sigmoid(X*theta);
J = -y'*log(h_x) - (1-y')*log(1-h_x);


% Gradient
err = sigmoid(X*theta) - y;

J_reg_term = 0;

for j = 1:columns(X)
    if (j > 1)
        % Cost regularization
        J_reg_term = J_reg_term + (lambda/(2))*theta(j)^2;
        % Gradient regularization
        grad(j) = sum(err.*X(:,j)) + lambda*theta(j);
    else
        % Gradient
        grad(j) = sum(err.*X(:,j));
    end  
end

grad = grad / m;

J = (J + J_reg_term) / m;



% =============================================================

end
