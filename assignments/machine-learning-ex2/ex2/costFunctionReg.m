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
z = X * theta;
errorSum = X' * (sigmoid(z) - y);
grad = errorSum / m;
thetaSquared_Sum = 0;

for position = 1:size(grad,1)
  if(position > 1)  grad(position) = grad(position) + (lambda / m) * theta(position);
                    thetaSquared_Sum = theta(position) ^ 2 + thetaSquared_Sum;
end

J = (-1 / m) * (y' * log(sigmoid(z)) + (1 - y)' * log(1 - sigmoid(z))) + (lambda / (2 * m))*thetaSquared_Sum;

% =============================================================

end
