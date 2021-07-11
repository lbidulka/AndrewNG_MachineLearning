function [C, sigma] = dataset3Params(X, y, Xval, yval)
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

C_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
sigma_vec = C_vec;

C_best = 0;
sigma_best = 0;


% Initialize the values
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
predictions = svmPredict(model, Xval);
err = mean(double(predictions ~= yval));
err_min = err;

% Get errors for each C and sigma
for i = 1:length(C_vec)
    C = C_vec(i);
    for j = 1:length(sigma_vec)
        sigma = sigma_vec(j);

        % Train and visualize the model
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
        %visualizeBoundary(X, y, model);

        % Compute the error for the chosen C and sigma
        predictions = svmPredict(model, Xval);
        err = mean(double(predictions ~= yval));

        % If this is the new lowest error, update the best C and sigma values
        if err < err_min
            err_min = err;
            C_best = C;
            sigma_best = sigma;
        end 
    end
end

disp("Optimal C and sigma: ")
C = C_best
sigma = sigma_best


% =========================================================================

end
