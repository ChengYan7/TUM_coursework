function features = FAST_detector(input_image, varargin)
  %% implement a Features From Accelerate Segment Test (FAST) detector that extracts features from the input_image
   % Features from accelerated segment test (FAST) is a corner detection method, 
   % which could be used to extract feature points and later used to track and map objects in many computer vision tasks. 
   % 
   %
  %% Input parser
   % tau                threshold value for detection of a corner
   % do_plot            image display variable
   % min_dist           minimal distance of two features in pixels
   % tile_size          size of the tiles
   % N                  maximal number of features per tile
   
   % Set the limit and default value of the function input
   P = inputParser;
   % tau                threshold value for detection of a corner
   defaultTau = double(60); 
   validScalarPosNum3 = @(x) isnumeric(x) && (x > 0)  && isa(x, 'double');
   addOptional(P,'tau',defaultTau, validScalarPosNum3);
   % do_plot            image display variable
   defaultD = logical(0);
   addOptional(P, 'do_plot', defaultD, @islogical)
   % min_dist           minimal distance of two features in pixels
   defaultmin_dist = 20;
   validScalarPosNum4 = @(x) isnumeric(x) && isscalar(x) && (x >= 1) ;
   addOptional(P,'min_dist', defaultmin_dist, validScalarPosNum4);
   % tile_size          size of the tiles
   defaulttile_size = [200,200];
   validScalarPosNum5 = @(x) isnumeric(x) && ismatrix(x)   ;
   addOptional(P,'tile_size', defaulttile_size, validScalarPosNum5);
   % N                  maximal number of features per tile
   defaultN = 5;
   validScalarPosNum6 = @(x) isnumeric(x) && isscalar(x) && (x >= 1) ;
   addOptional(P,'N', defaultN, validScalarPosNum6);
   % Parses the input in the function 
   parse(P, varargin{:});
   % output correspond parameter 
   tau = P.Results.tau
   do_plot = P.Results.do_plot
   min_dist = P.Results.min_dist
   tile_size = P.Results.tile_size
   if size(tile_size,2) == 1 
       tile_size = [tile_size, tile_size]
   end
   N = P.Results.N
      

   %% FAST corner detector 
    [m n]=size(input_image);
    corners=zeros(m,n);
    for i=4:m-3
        for j=4:n-3
            P=input_image(i,j);    
            % get 16 neighborhood points P_1-P_16 centered on P_0
            % These 16 points are in the shape of a circle with a radius of 3 pixels
            P_n=[input_image(i-3,j) input_image(i-3,j+1) input_image(i-2,j+2) input_image(i-1,j+3) ...
                input_image(i,j+3) input_image(i+1,j+3) input_image(i+2,j+2) input_image(i+3,j+1) ...
                input_image(i+3,j) input_image(i+3,j-1) input_image(i+2,j-2) input_image(i+1,j-3) ...
                input_image(i,j-3) input_image(i-1,j-3) input_image(i-2,j-2) input_image(i-3,j-1)];
            % 2. use threshold value filtering             
            % 2.1 calculate P1 and P9 (preprocess)
            if abs(P_n(1)-P) < tau && abs(P_n(9)-P) < tau
                continue; 
            end
            % 2.2 calculate P1,P9,P5,P13 (at least 3 points satisfied)   
            P_1_5_9_13=[abs(P_n(1)-P)>tau abs(P_n(5)-P)>tau abs(P_n(9)-P)>tau abs(P_n(13)-P)>tau];
            if sum(P_1_5_9_13)>=3
                ind=find(abs(P_n - P)>tau);
                % 2.3 calculate P1-P16 (at least 9 points satisfied)       
                if length(ind)>=9
                    corners(i,j) = sum(abs(P_n - P));      
                end
            end
        end
    end

      
    %% Index all features
    % sorted_index      sorted indices of features in decreasing order of feature strength
    
    % add a border to corners
    sz = size(corners);
    A = zeros(sz(1)+2*min_dist, sz(2)+2*min_dist);
    A(1+min_dist:min_dist+sz(1), 1+min_dist:min_dist+sz(2)) = corners;
    corners = A ;
    % corners = padarray(corners, [min_dist, min_dist])
    % A = corners;
    
    % sort the indices of all non-zero features in corners in decreasing
    corners = corners(:);
    [egal, sorted_index] = sort(corners, 'descend') ; % descend order
    num_non0 = nnz(corners)                           % number of non zero
    sorted_index = sorted_index(1:num_non0, :);
    corners = A;
    
    %% Feature preparation 
    % Feature detection with minimal distance and maximal number of features per tile
    % acc_array         accumulator array which counts the features per tile
    % features          empty array for storing the final features
    % N                 the upper limit of each tile
    
    % After dividing the input image into tiles, record the number of feature points in each block 
    acc_array = zeros([ceil(size(input_image,1)/tile_size(1)) ceil(size(input_image,2)/tile_size(2))]);
    % Record the feature points into the matrix (currently all of them are 0)
    features = zeros([ 2 min(numel(sorted_index), numel(acc_array)*N) ]);
    
    % number of new features
    n = 0;         
    % index to coordinate
    [y,x] = ind2sub(size(corners), sorted_index);          
    for i = 1:size(sorted_index, 1)
        % row in the order of tile
        x_acc = ceil((x(i)-min_dist)/tile_size(2)) ;    
        % column in the order of tile
        y_acc = ceil((y(i)-min_dist)/tile_size(1)) ;       
        if corners(y(i), x(i)) ~=0 && acc_array(y_acc, x_acc) < N
            % use cake function & mark the features' coordinate
            acc_array(y_acc, x_acc) = acc_array(y_acc, x_acc)+1;
            features(:,n+1) = [x(i)-min_dist; y(i)-min_dist];
            n = n + 1;
            corners(y(i)-min_dist:y(i)+min_dist, x(i)-min_dist:x(i)+min_dist) = ...
            corners(y(i)-min_dist:y(i)+min_dist, x(i)-min_dist:x(i)+min_dist).* cake(min_dist); 
        elseif corners(y(i), x(i)) ~=0 && acc_array(y_acc, x_acc) >= N
        % eliminate all the entried in this tile
            corners((((y_acc-1)*tile_size(1))+1+min_dist):min(size(corners, 1),y_acc*tile_size(1)+min_dist), ...
            (((x_acc-1)*tile_size(2))+1+min_dist):min(size(corners,2),x_acc*tile_size(2)+min_dist))=0;
        end
    end
    features = features(:, 1:n) ;
    X = features(1,:).';
    Y = features(2,:).';
    % Plot Routine
    im = cast(input_image,'uint8');
    figure
    imshow(im)
    hold on

    plot(X,Y, 'gs')
    
    
end