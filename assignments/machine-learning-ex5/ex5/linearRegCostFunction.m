function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% Calculate linear regression error
error = X * theta - y; %error for each data observation
squaredError = error .^2; %squared error for each data observation
squaredErrorSum = sum(squaredError); %sum the squared error of all data observations

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% For regularization part of the cost function formula use a temporary theta, where the first term is equal to zero so that we do not include theta zero into the regularization term
theta_temp = theta;
theta_temp(1) = 0;

% Calculate the cost function as per its formula
J = squaredErrorSum / (2 * m) + (lambda / (2 * m)) * sum(theta_temp .^ 2);

% Calculate the gradient for each theta term as per formula
grad = (X' * error) / m + (lambda / m) * theta_temp;

% =========================================================================

grad = grad(:);

end
