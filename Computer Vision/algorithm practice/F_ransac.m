function [correspondences_robust, largest_set_F] = F_ransac(correspondences, varargin)
    % This function implements the RANSAC algorithm to determine 
    % robust corresponding image points
    
    %% Input parser
    % Known variables:
    % epsilon       estimated probability
    % p             desired probability
    % tolerance     tolerance to belong to the consensus-set
    % x1_pixel      homogeneous pixel coordinates
    % x2_pixel      homogeneous pixel coordinates
    
    p = inputParser;
    % epsilon 50,0%
    defaultepsilon = double(0.5);         
    validScalarPosNum1 = @(x) isnumeric(x) && (x > 0) && (x < 1);
    addOptional(p,'epsilon',defaultepsilon, validScalarPosNum1);
    % p 50,0%
    defaultp = double(0.5);         
    validScalarPosNum2 = @(x) isnumeric(x) && (x > 0) && (x < 1);
    addOptional(p,'p',defaultp, validScalarPosNum2);
    % tolerance 0.01
    defaulttolerance = 0.01;         
    validScalarPosNum3 = @(x) isnumeric(x) && (x > 0) && (x < 1);
    addOptional(p,'tolerance',defaulttolerance, validScalarPosNum3);
    % Parses the input in the function 
    parse(p, varargin{:});
    % output correspond parameter 
    epsilon = p.Results.epsilon
    tolerance = p.Results.tolerance
    p = p.Results.p   

    % x1_pixel
    x1_pixel = [];
    x1_pixel(1:2,:) = correspondences(1:2,:);
    x1_pixel(3,:) = ones(1,size(x1_pixel,2));
    % x2_pixel
    x2_pixel = [];
    x2_pixel(1:2,:) = correspondences(3:4,:);
    x2_pixel(3,:) = ones(1,size(x2_pixel,2));
        
    %% RANSAC algorithm preparation
    % Pre-initialized variables:
    % k                     number of necessary points
    % s                     iteration number
    % largest_set_size      size of the so far biggest consensus-set
    % largest_set_dist      Sampson distance of the so far biggest consensus-set
    % largest_set_F         fundamental matrix of the so far biggest consensus-set
    
    k = 8;
    s = log(1-p)/log(1-(1-epsilon)^k);
    largest_set_size = zeros(1,1);
    largest_set_dist = inf*ones(1,1);
    largest_set_F = zeros(3,3);   
    
    %% RANSAC algorithm
    % correspondences_robust
    % largest_set_F

    %best_model = null              correspondences_robust
    %best_consensus_set = null      largest_set_F
    %best_error = infinity          largest_set_dist
    set_size = largest_set_size;
    set_dist = largest_set_dist;
    
    % perform each of these steps in every iteration i<=s
    for i = 1:s
        % 1. randomly choose corresponding image points
        random = randperm(size(correspondences,2)) ;
        correspondences_robust_can = [];
        for j = 1:k
            correspondences_robust_can(:,j) = correspondences(:,random(:,j));
        end
        % estimate the fundamental matrix
        F = epa(correspondences_robust_can);
        % 2. calsulate the Sampson distance for all corresponding image points
        s_dist = sampson_dist(F, x1_pixel, x2_pixel)
        % 3. if sd<tolerance, the corresponding points are includes in consensus set
        x1_pixel_robust = correspondences_robust_can(1:2,:);
        x1_pixel_robust(3,:) = ones(1, size(x1_pixel_robust,2));
        x2_pixel_robust = correspondences_robust_can(3:4,:);
        x2_pixel_robust(3,:) = ones(1, size(x2_pixel_robust,2));
        for m = 1:size(correspondences_robust_can,2)
            sd = sampson_dist(F, x1_pixel_robust, x2_pixel_robust)
            % 4. calculate the num of corr points and the sum of Sampson dist
            if sd < tolerance 
                set_size = set_size + 1 ; 
                set_dist = set_dist + sd ;
            else
                % mark the columns that are supposed to be delete                
                correspondences_robust_can(:,m) = zeros(4,1);
            end
        end
        % delete 0 column
        correspondences_robust_can(all(correspondences_robust_can==0,1))=[];
        % 5. & 6. 
        if set_size > largest_set_size
            largest_set_size = set_size;
            largest_set_dist = set_dist;
            largest_set_F = F;
            correspondences_robust = correspondences_robust_can; 
        elseif set_size == largest_set_size & set_dist < largest_set_dist
            largest_set_dist = set_dist;
            largest_set_F = F;
            correspondences_robust = correspondences_robust_can;
        end
    end
    
end