function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%%%Hard coded FFP algorithm
%%%Add a0(1)
a_1=[ones(m,1),X];

%%Compute a_2
a_2=sigmoid(a_1*Theta1');

%%Add a0_(2)
a_2=[ones(size(a_2,1),1),a_2];

%%%Compute h_thetaX
hX=sigmoid(a_2*Theta2');

%%%Matrix of y values
Y=eye(num_labels);

%%%Computing the unregularized cost 

for i=1:m
    for k=1:num_labels
    J=J+(-log(hX(i,k))*Y(k,y(i))-(1-Y(k,y(i)))*(log(1-hX(i,k))));
  end
end  

%%%%Computing the regularized cost 
J=(1/m)*J+(lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

  %%%Backpropagation algorithm
  
  %%%Initialize Accumulators 
  Delta_1=0;
  Delta_2=0;
  
  for i=1:m
  %%%Forward pass 
  %%Get the ith training example
  a1=X(i,:);

  %%Add bias unit a_0(1) 
  a1=[1,a1]';

  %%Get a_2
  a2=sigmoid(Theta1*a1);

  %%Add bias unit
  a2=[1;a2]; 

  %%Compute a_3 or hX
  h_X=sigmoid(Theta2*a2);

  %%%Back propagation 

    %%Compute delta_3
    delta_3=zeros(size(h_X,1),1); 
    
      for k=1:num_labels 
        delta_3(k)=h_X(k)-Y(k,y(i));
      end    
    
    %%%Compute delta_2
    delta_2=Theta2'*delta_3.*(a2.*(1-a2));
    
    %%%Update Accumulator 
    
    Delta_1=Delta_1 + delta_2(2:end)*a1';
    
    Delta_2=Delta_2 + delta_3*a2';
    
  end
 
 %%%Return Theta1 & Theta2 unregularized gradients 
 Theta1_grad=(1/m)*Delta_1;
 
 Theta2_grad=(1/m)*Delta_2;
    
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.


%%%%Set the first column of Theta1 & Theta2 to zero 

Theta1(:,1)=0;
Theta2(:,1)=0;

%%%Return Theta1 & Theta2 regularized gradients 

Theta1_grad=(1/m)*Delta_1+(lambda/(m))*Theta1;

Theta2_grad=(1/m)*Delta_2+(lambda/(m))*Theta2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
