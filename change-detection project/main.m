clearvars;
clc;
warning('off','all');
x = input('Do you want to run the app?(y/n)\n','s');
if isequal(x,'y')
    run('gui.mlapp');
else
    %% preparation for change detection
    % choose reference image
    disp('choose one reference image in .jpg format')
    [ref_name, ref_path]=uigetfile('*.jpg','select one file');
    [~,ref_time,~] = fileparts(ref_name);
    year1 = str2double(ref_time(1:4));
    month1 = str2double(ref_time(6:7));
    
    ref_image = imread(strcat(ref_path,ref_name));
    ref_gray = rgb2gray(ref_image);
    %disp('you have chosen the folder: ' fprintf ref_path)
    %fprintf(['you have chosen the folder: ' ref_path '\n'])
    disp(['you have chosen the folder: ' ref_path]);
    
    % choose comparing image. Now only one image is allowed
    disp('choose one comparing images in .jpg format in the same folder')
    [comp_name, comp_path]=uigetfile('*.jpg','select one file');
    
    % check if the choosen images are from the same subfolder
    if isequal(ref_path,comp_path)
    else
        error('ref_images and comp_images are not from the same subfolder');
    end
    
    tic
    % load comparing images
    [~,comp_time,~] = fileparts(comp_name);
    comp_image = imread(strcat(comp_path,comp_name));
    year2 = str2double(comp_time(1:4));
    month2 = str2double(comp_time(6:7));
    
    % check if the comparing images are the same as the reference image
    if isequal(ref_name,comp_name)
        warning('comp_image cannot be the same as the ref_image!');
    end
    % output the difference of 2 time (month)
    time_month = abs((12*year2+month2)-(12*year1+month1));
    %% image normalized
    normed_comp_image = normalized(comp_image,ref_image);
    
    figure(1);
    subplot(1,3,1),imshow(ref_image);
    title('Reference Image');
    subplot(1,3,2),imshow(comp_image);
    title('Target Image');
    subplot(1,3,3),imshow(normed_comp_image);
    title('Normalized Target Image');
    
    %% image regiatration
    comp_gray = rgb2gray(comp_image);
    threshold=0.0004;
    [comp_align_gray,comp_align_rgb] = alignment(comp_image,ref_image,threshold);
    
    %% differential image
    % pca_kmeans: output the differential image by using PCA and k-means
    pca_threshold=100;
    diff_im=pca_kmeans(ref_image,comp_align_rgb,pca_threshold,8);
    
    % plot result
    figure(2);
    subplot(1,3,1),imshow(ref_image);
    title('Reference Image');
    subplot(1,3,2),imshow(comp_align_rgb);
    title('Registiered Image');
    subplot(1,3,3),imshow(diff_im)
    title('Differential Image');
    hold off
    
    %% Detect the change information in images
    % we use white pixels to represent the change information
    % and black pixels to represent the unchanged information
    
    % Initialization
    [W,L,C] = size(diff_im);
    sum_pixels = W*L;
    number_EdgePixels=0;
    number_AlignedPixels=0;
    number_DifferentPixels=0;
    edge_pixels=zeros(W,L);
    different_pixels=zeros(W,L);
    same_pixels=zeros(W,L);
    
    % divide the part with different information in images:
    % edge_pixels:          unaligned area between images
    % different_pixels:     differential information in aligned area
    % same_pixels:          same information in aligned area
    image_align_gray = rgb2gray(comp_align_rgb);
    for i=1:W
        for j = 1:L
            if image_align_gray(i,j)==0
                edge_pixels(i,j)=1;
                number_EdgePixels=number_EdgePixels+1;
                diff_im(i,j)=0;
            elseif (image_align_gray(i,j)~=0)&&(diff_im(i,j)==1)
                different_pixels(i,j)=1;
                number_DifferentPixels=number_DifferentPixels+1;
            else
                same_pixels(i,j)=1;
            end
        end
    end
    
    % Assign the pixels of change information the value of 1
    number_AlignedPixels=sum_pixels-number_EdgePixels;
    number_SamePixels=number_AlignedPixels-number_DifferentPixels;
    if (number_DifferentPixels/number_AlignedPixels)>0.35
        different_pixels=same_pixels;
    end
    diff_im=different_pixels;
    
    %% Visualization the change information with highlight
    
    image_highlight = ref_image + uint8(cat(3,diff_im*1*255,diff_im*0*255,diff_im*0*255));
    figure(3);
    imshow(image_highlight);
    title('Highlight of Change Information');
    
    %% Change Information Analysis
    % we plan to reflect the magni-tude and speed of changes in the following metrics
    % ch_ratio:Change Ratio
    % ch_speed:Change Speed
    % isUrban:Location Classification
    [ch_ratio,ch_speed] = change_analysis(number_AlignedPixels,number_DifferentPixels,time_month);
    fprintf([ 'Changing Time: ' num2str(time_month,'%04d') ' months\n' ...
        'Changing ratio: ' num2str(ch_ratio,'%.2f') '\n' ...
        'Change speed: ' num2str(ch_speed,'%.2f') '\n'...
        ]);
    [highlight_nature,highlight_urban,isUrban] = classification(comp_image);
    if isUrban == 1
        disp('This changed area is urban');
    else
        disp('This changed area is nature');
    end
    
    figure(4)
    subplot(1,2,1);
    imshow(highlight_urban);
    title('Highlight of Urban Information');
    subplot(1,2,2);
    imshow(highlight_nature);
    title('Highlight of Nature Information');
    toc
end

