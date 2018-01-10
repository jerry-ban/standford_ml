function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%



theta_x = X*theta;
h_x = sigmoid(theta_x);
prob_x =log( h_x );
prob_x1 = log( 1-h_x);

cs = size(y,1);
n = length(grad);

%delta_y = - y * prob_x - (1-y)' * prob_x1;
%theta_v = theta([2:length(theta)],1);
%J = J + delta_y / m + lambda * sum( theta_v.^2) / (2*m);

delta_y = - y' * prob_x - (1-y)' * prob_x1;
theta_v = theta([2:length(theta)],1);
J = sum(delta_y / m + lambda * sum( theta_v.^2) / (2*m) );  


  y_cs= y ;%(cs_i,1);
   
  for i_n = 1: 1
    grad(i_n) = grad(i_n) + (h_x - y_cs)' * X(:, i_n) / m;
  end

  for i_n = 2: n
      grad(i_n) = grad(i_n) + (h_x - y_cs)' * X(:, i_n) / m + lambda * theta(i_n) / m;
  end




% =============================================================

end
