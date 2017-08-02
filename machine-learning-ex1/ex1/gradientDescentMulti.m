% function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
% %GRADIENTDESCENTMULTI Performs gradient descent to learn theta
% %   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
% %   taking num_iters gradient steps with learning rate alpha
% 
% % Initialize some useful values
% m = length(y); % number of training examples
% J_history = zeros(num_iters, 1);
% 
% for iter = 1:num_iters
% 
%     % ====================== YOUR CODE HERE ======================
%     % Instructions: Perform a single gradient step on the parameter vector
%     %               theta. 
%     %
%     % Hint: While debugging, it can be useful to print out the values
%     %       of the cost function (computeCostMulti) and gradient here.
%     %
%     summ = zeros(3,1);
%     temp = zeros(3,1);
%     
%     for i = 1:m
%         summ(1) = summ(1) + (theta' * X(i,:)' - y(i)) * X(i,1);
%         summ(2) = summ(2) + (theta' * X(i,:)' - y(i)) * X(i,2);
%         summ(3) = summ(3) + (theta' * X(i,:)' - y(i)) * X(i,3);
%     end
%     
%     dJ_dt(1) = 1/m * summ(1);
%     dJ_dt(2) = 1/m * summ(2);
%     dJ_dt(3) = 1/m * summ(3);
%     
%     temp(1) = theta(1) - alpha * dJ_dt(1);
%     temp(2) = theta(2) - alpha * dJ_dt(2);
%     temp(3) = theta(3) - alpha * dJ_dt(3);
%     
%     theta(1) = temp(1);
%     theta(2) = temp(2);
%     theta(3) = temp(3);
%     
%     % ============================================================
% 
%     % Save the cost J in every iteration    
%     J_history(iter) = computeCostMulti(X, y, theta);
% 
% end
% 
% end

function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

	A = X * theta - y;  % (m x 1 vector)
	delta = 1 / m * (A' * X)';  % ' ((n+1) x 1 vector)
	theta = theta - (alpha * delta); % ' ((n+1) x 1 vector)

    % ============================================================

    % Save the cost J in every iteration    
	J_history(iter) = computeCostMulti(X, y, theta);

end

end