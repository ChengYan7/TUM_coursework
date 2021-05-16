function features = harris_detector(input_image, varargin)
   % implement a Harris detector that extracts features from the input_image

    
  %% Input parser from task 1.7
   % segment_length    size of the image segment
   % k                 weighting between corner- and edge-priority
   % tau               threshold value for detection of a corner
   % do_plot           image display variable
   % min_dist          minimal distance of two features in pixels
   % tile_size         size of the tiles
   % N                 maximal number of features per tile
   %设置函数输入量的限制条件和默认值
   p = inputParser;
   
   % 设置 segment_length      size of the image segment
   defaultS = double(15);         
   validScalarPosNum1 = @(x) isnumeric(x) && (x > 1) && (mod(x,2) == 1) && isa(x, 'double');
   addOptional(p,'segment_length',defaultS, validScalarPosNum1);
   
   % 设置 k                   weighting between corner- and edge-priority
   defaultK = 0.05;
   validScalarPosNum2 = @(x) isnumeric(x) && isscalar(x) && (x >= 0) && (x <= 1) ;
   addOptional(p,'k', defaultK, validScalarPosNum2);
   
   % 设置 tau                 threshold value for detection of a corner
   defaultTau = double(10^6); 
   validScalarPosNum3 = @(x) isnumeric(x) && (x > 0)  && isa(x, 'double');
   addOptional(p,'tau',defaultTau, validScalarPosNum3);
   
   % 设置 do_plot             image display variable
   defaultD = logical(0);
   addOptional(p, 'do_plot', defaultD, @islogical)
   
   % 设置 min_dist            minimal distance of two features in pixels
   defaultmin_dist = 20;
   validScalarPosNum4 = @(x) isnumeric(x) && isscalar(x) && (x >= 1) ;
   addOptional(p,'min_dist', defaultmin_dist, validScalarPosNum4);
   
   % 设置 tile_size           size of the tiles
   defaulttile_size = [200,200];
   validScalarPosNum5 = @(x) isnumeric(x) && ismatrix(x)   ;
   addOptional(p,'tile_size', defaulttile_size, validScalarPosNum5);
   
   % 设置 N                   maximal number of features per tile
   defaultN = 5;%5
   validScalarPosNum6 = @(x) isnumeric(x) && isscalar(x) && (x >= 1) ;
   addOptional(p,'N', defaultN, validScalarPosNum6);
   
   % Parses the input in the function 
   parse(p, varargin{:});
   
   % output correspond parameter 
   segment_length = p.Results.segment_length
   k = p.Results.k
   tau = p.Results.tau
   do_plot = p.Results.do_plot
   min_dist = p.Results.min_dist
   tile_size = p.Results.tile_size
   if size(tile_size,2) == 1 
       tile_size = [tile_size, tile_size]
   end
   
   N = p.Results.N
      
         
   %% Preparation for feature extraction
    % Ix, Iy            image gradient in x- and y-direction
    % w                 weighting vector
    % G11, G12, G22     entries of the Harris matrix
    % Check if it is a grayscale image
    image_size = size(input_image);
    dimension = numel(image_size);
    if dimension == 2
    else
        error('Image format has to be NxMx1');
    end
    im = double(input_image);
    
    % Approximation of the image gradient
    [Ix, Iy] = sobel_xy(im);
    
    % 加权滤波 Weighting
    % segment_length = 5×sigma
    sigma = 1/5*segment_length;          
    w = fspecial('gaussian', [1,segment_length], sigma);
    % W = w*w'
    W = fspecial('gaussian', [segment_length,segment_length], sigma);
    
    % Harris Matrix G
    Ix2=Ix.^2;
    Iy2=Iy.^2;
    Ixy=Ix.*Iy;
    G11 = filter2(W, Ix2);
    G22 = filter2(W, Iy2);
    G12 = filter2(W, Ixy);
        
    
   %% Feature extraction with the Harris measurement
    % corners           matrix containing the value of the Harris measurement for each pixel 
    %计算每个像元的Harris响应值
    [height,width]=size(im);
    H=zeros(height,width);
    
    %像素(i,j)处的Harris响应值
    for i=1:height
        for j=1:width
            M=[G11(i,j) G12(i,j);G12(i,j) G22(i,j)];
            H(i,j)=det(M)-k*(trace(M))^2;
        end
    end
    corners = H;
    
    % eliminate all features which are smaller than the threshold value tau
    % 删去小于阈值的特征点
    for i = 1:height
        for j = 1:width
            if corners(i,j) < tau
                 corners(i,j) = 0;
            end
        end
    end
    % 3×3领域非极大值抑制，极值置为1，其余置为0
    %corners_ = imregionalmax(corners);
    %corners_ = logical(corners);
    %[y,x] = find(corners_ == 1);
    %features = [x,y].'
     
    
   %% Feature preparation
    % sorted_index      sorted indices of features in decreasing order of feature strength
    % add a border to corners
    
    %将corners周围加上“最小距离”的边框（便于后面乘Cake矩阵）
    %corners = padarray(corners, [min_dist, min_dist])
    %A = corners;
    sz = size(corners);
    A = zeros(sz(1)+2*min_dist, sz(2)+2*min_dist);
    A(1+min_dist:min_dist+sz(1), 1+min_dist:min_dist+sz(2)) = corners;
    corners = A;
    
    %sort the indices of all non-zero features in corners in decreasing
    %将corners中元素按大小降序排列并生成索引在sorted_index中
    corners = corners(:);
    [egal, sorted_index] = sort(corners, 'descend') ; % descend order
    num_non0 = nnz(corners)                           % number of non zero
    sorted_index = sorted_index(1:num_non0, :);
    corners = A;
          
    
   %% Accumulator array
    % acc_array         accumulator array which counts the features per tile
    % features          empty array for storing the final features
    %将输入图像分块tile后，通过acc_array矩阵记录每一块的特征点数量（上限为N）
    acc_array = zeros([ceil(size(input_image,1)/tile_size(1)) ceil(size(input_image,2)/tile_size(2))]);
    %将特征点记录进矩阵（目前全是0）
    features = zeros([ 2 min(numel(sorted_index), numel(acc_array)*N) ]);
    
    
   %% Feature detection with minimal distance and maximal number of features per tile
     n = 0;          % number of new features
    [y,x] = ind2sub(size(corners), sorted_index);           % index to coordinate
    for i = 1:size(sorted_index, 1)
        x_acc = ceil((x(i)-min_dist)/tile_size(2)) ;       % row in the order of tile
        y_acc = ceil((y(i)-min_dist)/tile_size(1)) ;       % column in the order of tile
        if corners(y(i), x(i)) ~=0 && acc_array(y_acc, x_acc) < N
            % use cake function & mark the features' coordinate
            acc_array(y_acc, x_acc) = acc_array(y_acc, x_acc)+1;
            features(:,n+1) = [x(i)-min_dist; y(i)-min_dist];
            n = n + 1;
            corners(y(i)-min_dist:y(i)+min_dist, x(i)-min_dist:x(i)+min_dist) = corners(y(i)-min_dist:y(i)+min_dist, x(i)-min_dist:x(i)+min_dist).* cake(min_dist); 
        elseif corners(y(i), x(i)) ~=0 && acc_array(y_acc, x_acc) >= N
        % eliminate all the entried in this tile
            corners((((y_acc-1)*tile_size(1))+1+min_dist):min(size(corners, 1),y_acc*tile_size(1)+min_dist),(((x_acc-1)*tile_size(2))+1+min_dist):min(size(corners,2),x_acc*tile_size(2)+min_dist))=0;
        end
    end
    features = features(:, 1:n) ;
    X = features(1,:).';
    Y = features(2,:).';
    % Plot Routine
    im = cast(input_image,'uint8');
    imshow(im)
    hold on
    plot(X,Y, 'gs')
    
    
end