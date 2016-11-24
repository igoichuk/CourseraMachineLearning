function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples

% Compute J the cost of a particular choice of theta
Xt = X*theta;
J = sum(-y.*log(sigmoid(Xt)) - (1-y).*log(1-sigmoid(Xt)))/m + lambda*sum(theta(2:end).^2)/(2*m);

% Compute the partial derivatives
grad = zeros(size(theta));

df = sigmoid(Xt) - y;

grad(1) = sum(df)/m;

for i=2:length(theta)
  grad(i) = sum(df.*X(:,i))/m + lambda/m*theta(i);
endfor

end
