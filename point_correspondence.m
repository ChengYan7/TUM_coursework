function cor = point_correspondence(I1, I2, Ftp1, Ftp2, varargin)
    % In this function you are going to compare the extracted features of a stereo recording
    % with NCC to determine corresponding image points.
    
    %% Input parser
    % window_length     side length of quadratic window
    % min_corr          threshold for the correlation of two features
    % do_plot           image display variable
    % Im1, Im2          input images (double)
    p = inputParser;
   
    % window_length
    defaultwindow_length = double(25);
    validScalarPosNum1 = @(x) isnumeric(x) && (x > 1) && (mod(x,2) == 1) && isa(x, 'double');
    addOptional(p,'window_length',defaultwindow_length, validScalarPosNum1);
   
    % min_corr
    defaultmin_corr = 0.95;
    validScalarPosNum2 = @(x) isnumeric(x) && isscalar(x) && (x > 0) && (x < 1) ;
    addOptional(p,'min_corr', defaultmin_corr, validScalarPosNum2);
   
    % do_plot
    defaultdo_plot = logical(0);
    addOptional(p, 'do_plot', defaultdo_plot, @islogical);
   
    % Parses the input in the function 
    parse(p, varargin{:});
   
    % output correspond parameter 
    window_length = p.Results.window_length;
    min_corr = p.Results.min_corr;
    do_plot = p.Results.do_plot;
 
    % Im1 Im2
    Im1 = double(I1); 
    Im2 = double(I2);
   

    %% Feature preparation
    % no_pts1, no_pts 2     number of features remaining in each image
    % Ftp1, Ftp2            preprocessed features
    
    % Im1
    for i = 1:size(Ftp1, 2)
        % delete features near left border
        if Ftp1(1, i) <= window_length/2
            Ftp1(:,i) = 0;
        % right border
        elseif Ftp1(1, i) >= size(Im1,2) - window_length/2 + 1
            Ftp1(:,i) = 0;
        % upper border    
        elseif Ftp1(2, i) <= window_length/2
            Ftp1(:,i) = 0;
        % lower border
        elseif Ftp1(2, i) >= size(Im1,1) - window_length/2 + 1
            Ftp1(:,i) = 0; 
        end
    end
    % delete all zero column
    Ftp1(:,all(Ftp1==0,1)) = [];
    no_pts1 = size(Ftp1, 2);
    
    % Im2
    for i = 1:size(Ftp2, 2)
        % left border
        if Ftp2(1, i) <= window_length/2
            Ftp2(:,i) = 0;
        % right border
        elseif Ftp2(1, i) >= size(Im1,2) - window_length/2 + 1
            Ftp2(:,i) = 0;
        % upper border
        elseif Ftp2(2, i) <= window_length/2
            Ftp2(:,i) = 0;
        % lower border
        elseif Ftp2(2, i) >= size(Im1,1) - window_length/2 + 1
            Ftp2(:,i) = 0; 
        end
    end
    % delete all zero column
    Ftp2(:,all(Ftp2==0,1)) = [];
    no_pts2 = size(Ftp2, 2);
    
    
    %% Normalization
    % Mat_feat_1            normalized windows in image 1
    % Mat_feat_2            normalized windows in image 2
    
    I1 = im2double(I1);
    I2 = im2double(I2);
    % window by window
    % construct Mat_feat_1
    Mat_feat_1 = zeros(window_length^2, size(Ftp1,2));
    for i = 1 : size(Ftp1, 2)
        x = Ftp1(1, i);
        y = Ftp1(2, i);
        win = I1(y-(window_length-1)/2 : y+(window_length-1)/2, x-(window_length-1)/2 : x+(window_length-1)/2);
        mean_win = mean(mean(win));
        r_win = reshape(win, size(win,1)*size(win,2), 1);
        std_win = std(r_win);
        win = win - mean_win*ones(window_length, window_length);
        W = win/std_win;
        Mat_feat_1(:, i) = reshape(W, size(W,1)*size(W,2), 1);
    end
    % construct Mat_feat_2
    Mat_feat_2 = zeros(window_length^2, size(Ftp2,2));
    for i = 1 : size(Ftp2, 2)
        x = Ftp2(1, i);
        y = Ftp2(2, i);
        win = I2(y-(window_length-1)/2 : y+(window_length-1)/2, x-(window_length-1)/2 : x+(window_length-1)/2);
        mean_win = mean(mean(win));
        r_win = reshape(win, size(win,1)*size(win,2), 1);
        std_win = std(r_win);
        win = win - mean_win*ones(window_length, window_length);
        W = win/std_win;
        Mat_feat_2(:, i) = reshape(W, size(W,1)*size(W,2), 1);
    end
    
   
    %% NCC calculations
    % NCC_matrix            matrix containing the correlation between the image points
    % sorted_index          sorted indices of NCC_matrix entries in decreasing order of intensity

    N = size(Mat_feat_1,1);
    NCC_matrix = 1/(N-1)*Mat_feat_2'*Mat_feat_1;    
    NCC_matrix(NCC_matrix < min_corr) = 0;
    % sort the indices of all non-zero enties in NCC_matrix in decreasing
    NCC = NCC_matrix(:);
    [egal, sorted_index] = sort(NCC, 'descend') ; % descend order
    num_non0 = nnz(NCC);                          % number of non zero
    sorted_index = sorted_index(1:num_non0, :);
    
    
    %% Correspondeces
    % cor                   matrix containing all corresponding image points
    
    cor = [];
    [y,x] = ind2sub(size(NCC_matrix), sorted_index);
    % if a correpondence point is found in a column, this column should be set to zero    
    % This procedure makes sure no feature in image 1 is mapped on more than one feature in image 2
    for i =1:numel(sorted_index)
        if NCC_matrix(sorted_index(i))
        % y represent the window in column y of Mat_feat_2, which is relavent to the y-th feature in Ftp2
        % x represent the window in column x of Mat_feat_1, which is relavent to the x-th feature in Ftp1
            fea = [Ftp1(:,x(i));Ftp2(:,y(i))];
            cor = [cor fea];
            NCC_matrix(:,x(i)) = 0;
        end
    end 
   
    
   %% Visualize the correspoinding image point pairs
    % features in image 2
    corI2 = cor(3:4,:)';
    % superposition of 2 images with a transparency of 50% 
    super = 1/2*I1+1/2*I2;
    imshow(super); 
    hold on
    for i = 1:size(cor,2)
        % plot features in image 1
        plot(corI1(i,2),corI1(i,1),'bs');
        % plot features in image 2
        plot(corI2(i,2),corI2(i,1),'ro');
        % plot line singly
        plot([corI1(i,2),corI2(i,2)], [corI1(i,1),corI2(i,1)],'g');
    end
    
    
end