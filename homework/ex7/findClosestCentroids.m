function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

m = size(X, 1);
X2 = sum( X.^2, 2);
centroids2 = sum(centroids.^2,2);
I_m = ones(m,1);
I_k = ones(K, 1);

sim = bsxfun(@plus, X2 * I_k', bsxfun(@plus, I_m * centroids2', - 2 * (X * centroids')));
min_set = min(sim, [], 2);
sim_I = sim == min_set;

for i = 1:m
  idx(i) = find(sim_I(i,:) == 1,1);
end

% =============================================================

end

