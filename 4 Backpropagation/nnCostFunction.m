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
Theta1_grad = zeros(size(Theta1)); % 25 x 401
Theta2_grad = zeros(size(Theta2)); % 10 x 26

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% Calculate A2 (A1 is X)
X = [ones(m, 1) X]; % add bias (5000 x 401)
A2 = sigmoid(X * Theta1'); % 5000 x 25
% Calculate A3 (result H) 
A2 = [ones(m, 1) A2]; % add bias (5000 x 26)
H = sigmoid(A2 * Theta2'); % 5000 x 10

% Convert vestion y into matrix Y where each row is K-size vector
Y = zeros(m, num_labels);
for i = 1:m
  Y(i, y(i)) = 1;
end

% Cost function
J = sum(sum(-Y.*log(H) - (1-Y).*log(1-H)))/m;

% Add regularization
J = J + lambda/(2*m)*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))); 
%J = J + lambda/(2*m)*(sum(sum(Theta1.^2)(2:end)) + sum(sum(Theta2.^2)(2:end))); 

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

for i = 1:m
  d3 = H(i,:) - Y(i,:); % row vector of 10
  a2 = A2(i,:); % row vector of 26
  d2 = (d3*Theta2.*a2.*(1 - a2))(2:end); % row vector of 25
  
  Theta2_grad = Theta2_grad + d3'*a2;
  Theta1_grad = Theta1_grad + d2'*X(i,:);
end

Theta2_grad = Theta2_grad/m;
Theta1_grad = Theta1_grad/m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

temp = Theta2;
temp(:,1) = 0; % nillify first column
Theta2_grad = Theta2_grad + lambda/m*temp;

temp = Theta1;
temp(:,1) = 0; % nillify first column
Theta1_grad = Theta1_grad + lambda/m*temp;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
