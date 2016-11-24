function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% Compute J the cost of a particular choice of theta
Xt = X*theta;
J = sum(-y.*log(sigmoid(Xt)) - (1-y).*log(1-sigmoid(Xt)))/m;

% Compute the partial derivatives
grad = zeros(size(theta));

df = sigmoid(Xt) - y;

for i=1:length(theta)
  grad(i) = sum(df.*X(:,i))/m;
endfor

end
