function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C_vec=[0.01;0.03;0.1;0.3;1;3;10;30];
sigma_vec=C_vec;

size_C_Vec=size(C_vec,1);
size_sigma_Vec=size(sigma_vec,1);

N=size_C_Vec*size_sigma_Vec;


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

%%%Matrix R is of dimensions (size_C_vec*size_sigma_vec) x 3 where the first 2 columns contain
%%%every possible combination of C and sigma values and where the 3rd column contains 
%%%the associated cost for Jval 

%%% C  sigma   %%%Jval
R=zeros(N,3); 
cnt=0; %%Counter used to control the row index of vector R

for i=1:size_C_Vec
  R(cnt+1:size_C_Vec*i,1)=C_vec(i);
  
  for j=1:size_sigma_Vec
    cnt=cnt+1;
    R(cnt,2)=sigma_vec(j);
  end
end

%%%%Compute Jval for each combination of C and sigma

for k=1:N
  C=R(k,1);
  sigma=R(k,2);
  
  %%%Compute predictions based on model in SVM train
  model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));  
  
  predictions = svmPredict(model, Xval);
  
  %%%Compute Jval or prediction error
  R(k,3)=mean(double(predictions ~= yval));
end

%%%Find the values of C and sigma that minimize prediction error
[min_Jval,idx]= min(R(:,3)); 
 
%%%Return values 
C=R(idx,1);
sigma=R(idx,2);
 
% =========================================================================

end
