function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
n = length(theta);
for iter = 1:num_iters
  theta_new = zeros(n,1);
  for iv = 1: n
    pred = X * theta;
    delta_y = pred - y;
    theta_new(iv) = theta(iv) - alpha / m * sum( delta_y .* X(:, iv) );
  end
  
  theta = theta_new;
  J_history(iter) = computeCostMulti(X, y, theta);
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %




    % ============================================================

    % Save the cost J in every iteration    

end

end
