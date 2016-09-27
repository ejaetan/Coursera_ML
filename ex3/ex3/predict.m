function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

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

% fprintf("size X: "), disp(size(X)); 5000 x 400
% fprintf("size Theta1: "), disp(size(Theta1)); 25 x 401
% fprintf("size Theta2: "), disp(size(Theta2)); 10 x 26

x = [ones(m,1),X];	% add bias values, x = 5000 x 401
a2 = sigmoid(x * Theta1');
% fprintf("size a2: "), disp(size(a2)); 5000 x 25
a2 = [ones(m,1),a2];	% 5000 x 26
a3 = sigmoid(a2 * Theta2');

% For two matrices (or a matrix and a scalar), return the pairwise maximum.
[M, p] = max(a3,[],2);






% =========================================================================


end
