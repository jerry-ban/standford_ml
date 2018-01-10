function [C, sigma,error_list] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

Cs = [0.01; 0.03; 0.1;0.3;1;3;10;10];
Sigmas =  Cs;
leng= size(Cs,1);
cs_list = zeros(leng^2, 1);
sigma_list = zeros(size(Cs,2)^2, 1);
error_list = zeros(size(Cs,2)^2, 1);
row_count = 0;
for c_i = 1:leng
  for sig_i = 1: leng
    row_count =row_count+1;
    %fprintf('try: %i \n', row_count);
    
    C= Cs(c_i);
    sigma = Sigmas(sig_i);
    
    cs_list(row_count) = C;
    sigma_list(row_count)= sigma;    
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
    predY = svmPredict(model, Xval);
    error_list(row_count) = mean(double(predY!=yval));
  end
end
min_set = find(error_list == min(error_list));
C = cs_list(min_set)(1);
sigma = sigma_list(min_set)(1);

% =========================================================================

end
