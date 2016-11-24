function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
% For reglarization
temp = theta;
temp(1) = 0; % because we don't regilarize theta_0

% Compute J the cost of a particular choice of theta
h = sigmoid(X*theta);
J = sum(-y.*log(h) - (1-y).*log(1-h))/m + lambda*sum(temp.^2)/(2*m); 

% Compute the partial derivatives
grad = X'*(h-y)/m; % unregularized gradient for logistic regression
grad = grad + lambda/m*temp; % regularization

% return column vector
%grad = grad(:);

end
