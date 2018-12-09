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

% Add ones to the X data matrix
a1 = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Convert the 5000x401 a1 input (added 1 for the zero parameter to the X input) into a 5000x25 input for layer 2 (i.e.,
% reduce dimensionality of raw input X) with the use of Theta1 (weights for layer 1). 
% Note that a1 contains the activation values (401 for each observation).
z2 = a1 * Theta1';

% Convert z2 into a2 using the sigmoid function so that the activation values are within the range 0-1.
a2 = sigmoid(z2);

% Add a column with 1's at the beginning of the a2 (input_layer2) matrix for the zero parameter.
a2 = [ones(m, 1) a2];

% Using Theta 2 (weights for layer 2), convert the 5000x26 input to a 5000x10 matrix, which 
% represents the output of the neural network. The values should be converted via the sigmoid function
% to values in the 0-1 range.
h = sigmoid(a2 * Theta2');

% Loop through all examples in the training dataset
for i = 1:m

% Take the i-th example from the training dataset (i-th row)
t = h(i,:);

% Create a num_labels-dimensional vector that represents the real value of the result (y); e.g., 10 possible outcome => 10-dimensional vector representing the real outcome
w = 1:num_labels;
% All vector values are zero except the value at the position in the vector that corresponds to the value of the outcome (y)
w = (w == y(i));

% Calculate the cost function (modified so that it correctly matches the dimensions of the respective vectors (rows and columns)
J = J + (-1 / m) * (log(t) * w' + log(1 - t) * (1 - w)');

end

% Transform the weight matrices (theta's) so that their first columns contain only zero values. This will exclude the bias terms of each weight matrix from the regularization process.
temp1 = Theta1;
temp1(:,1) = 0;
temp2 = Theta2;
temp2(:,1) = 0;

% Add regularization to the already computed cost function
J = J + (lambda / (2 * m)) * (sum(sum(temp1.^2)) + sum(sum(temp2.^2)))
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

% Loop through all examples in the training dataset
for i = 1:m

% Take the i-th example from the training dataset (i-th row)
t = h(i,:)';

% Take the i-th a1 vector element from the a1 matrix
a_1 = a1(i,:)';

% Take the i-th a2 vector element from the a2 matrix
a_2 = a2(i,:)';

% Create a num_labels-dimensional vector that represents the real value of the result (y); e.g., 10 possible outcome => 10-dimensional vector representing the real outcome
w = 1:num_labels;
% All vector values are zero except the value at the position in the vector that corresponds to the value of the outcome (y)
w = (w == y(i))';

% Create delta between predicted vector (t) and real vector (y in its vectorized form w)
delta_3 = t - w;

delta_2 = (Theta2)' * delta_3 .* (a_2 .* (1 - a_2));  % delta_3 * Theta2 .* (a_2 .* (1 - a_2));

% Remove the bias unit delta from delta_2
delta_2 = delta_2(2:end);

Theta2_grad = Theta2_grad + delta_3 .* a_2';
Theta1_grad = Theta1_grad + delta_2 .* a_1';

end

Theta2_grad = Theta2_grad / m;
Theta1_grad = Theta1_grad / m;

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Modify the weight matrices to correspond to the regularization formula (i.e., multiply by (lambda / m) and remove the weights of the bias unit from the matrices because they 
% should not be added when computing the gradients with regularization).
% This also computes the gradients for the regularization part of the gradient formula:
Theta1_temp = (lambda / m) * Theta1;
Theta1_temp(:,1) = 0;

Theta2_temp = (lambda / m) * Theta2;
Theta2_temp(:,1) = 0;

% Calculate the (total) gradients (deltas) for each position within each weight matrix (Theta)
Theta2_grad = Theta2_grad + Theta2_temp;
Theta1_grad = Theta1_grad + Theta1_temp;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
