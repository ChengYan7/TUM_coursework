function [EF] = eight_point_algorithm(correspondences, K)
    % Depending on whether a calibrating matrix 'K' is given,
    % this function calculates either the essential or the fundamental matrix
    % with the eight-point algorithm.
    %% First step of the eight-point algorithm from task 3.1
    % Known variables:
    % x1, x2        homogeneous (calibrated) coordinates       
    % A             matrix A for the eight-point algorithm
    % V             right-singular vectors
    
if nargin == 1      % no K input
    x1 = [];
    x1(1:2,:) = correspondences(1:2,:);
    x1(3,:) = ones(1,size(x1,2));
    x2 = [];
    x2(1:2,:) = correspondences(3:4,:);
    x2(3,:) = ones(1,size(x2,2));
    for i = 1:size(x1,2)
        A(i,:) = kron(x1(:,i),x2(:,i))';   
    end    
    [~,~,V] = svd(A);        
end
if nargin == 2      % with K input
    x1 = [];
    x1(1:2,:) = correspondences(1:2,:);
    x1(3,:) = ones(1,size(x1,2));
    x1 = inv(K)*x1;
    x2 = [];
    x2(1:2,:) = correspondences(3:4,:);
    x2(3,:) = ones(1,size(x2,2));
    x2 = inv(K)*x2;
    for i = 1:size(x1,2)
        A(i,:) = kron(x1(:,i),x2(:,i))';   
    end    
    [~,~,V] = svd(A); 
end

    %% Estimation of the matrices
if nargin == 1      % no K input    
    G = reshape(V(:,end),3,3);
    [U,Sigma,V] = svd(G);
    Sigma(3,3) = 0;
    EF = U * Sigma * V' ;    
end
if nargin == 2      % with K input
    G = reshape(V(:,end),3,3);
    [U,Sigma,V] = svd(G);
    Sigma_ = [1,0,0;0,1,0;0,0,0];
    EF = U * Sigma_ * V' ;    
end    

end