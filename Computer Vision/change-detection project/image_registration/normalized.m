function normedpic = normalized(X,ST)
% This program is used to normalize the photo brightness, adjusting photos of different brightness to the same brightness as the base photo. It is used to replace the histogram equalization function.
% 1. Select a photo with the best brightness condition as the reference, and find out the grayscale average mean_st of all pixels in V-space of this photo.
% 2. Find the grayscale average mean1 of all pixels in the V-space of the processed photo.
% 3. Add mean_st and subtract mean1 for each pixel in the V-space of the processed photo.
% X =  target image
% ST =  reference image
% transform the image from RGB space to HSV space
Ist=rgb2hsv(ST);
I1=rgb2hsv(X);
Hst=Ist(:,:,1);
H1=I1(:,:,1);
S1=I1(:,:,2);
Sst=Ist(:,:,2);
Vst=Ist(:,:,3);
V1=I1(:,:,3);
num_st=size(Vst);
mean_st=sum((sum(Vst))')/(num_st(1)*num_st(2));
num1=size(V1);
mean1=sum((sum(V1))')/(num1(1)*num1(2));
delta=mean_st-mean1;
k=-0.07;      % This needs to be set  parameter as you wish
temp=V1+delta+k;
HSV3(:,:,1)=H1(:,:);       % Keep H constant and start synthesis
HSV3(:,:,2)=S1(:,:);
HSV3(:,:,3)=temp(:,:);
normedpic=hsv2rgb(HSV3);    % transformation back to RGB space

end