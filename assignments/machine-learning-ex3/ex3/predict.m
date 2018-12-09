function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
% Convert the 5000x401 X input (added 1 for the zero parameter) into a 5000x25 input for layer 2 (i.e.,
% reduce dimensionality of raw input X) with the use of Theta1 (weights for layer 1). 
% Note that X contains the activation values (401 for each observation).
z2 = X * Theta1';

% Convert z2 into a2 using the sigmoid function so that the activation values are within the range 0-1.
a2 = sigmoid(z2);

% Add a column with 1's at the beginning of the a2 (input_layer2) matrix for the zero parameter.
a2 = [ones(m, 1) a2];

% Using Theta 2 (weights for layer 2), convert the 5000x26 input to a 5000x10 matrix, which 
% represents the output of the neural network. The values should be converted via the sigmoid function
% to values in the 0-1 range.
h = sigmoid(a2 * Theta2');

% Determine the label with the largest output value of each example in the training set (observation).
% Take its index in the matrix along the row (imax_prob).
[max_output imax_output] = max(h, [], 2);

% Assign index of max output to p.
p = imax_output;

% =========================================================================


end
