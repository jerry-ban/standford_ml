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




theta_x = X*theta;
h_x = sigmoid(theta_x);
prob_x =log( h_x );
prob_x1 = log( 1-h_x);
delta_y = - y' * prob_x - (1-y)' * prob_x1;
theta_v = theta([2:length(theta)],1);
J = delta_y / m + lambda * sum( theta_v.^2) / (2*m);

n = length(grad);

for i_n = 1: 1
  grad(i_n) = (h_x - y)' * X(:, i_n) / m;
end

for i_n = 2: n
  grad(i_n) = (h_x - y)' * X(:, i_n) / m + lambda * theta(i_n) / m;
end



% =============================================================

end
