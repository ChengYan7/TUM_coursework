function [image_di,number_AlignedPixels,number_DifferentPixels] = differential(image_ref,image_align,threshold)
image_di = pca_kmeans(image_ref,image_align,threshold,8);
[W,L,~] = size(image_di);
sum_pixels = W*L;
number_EdgePixels=0;
number_DifferentPixels=0;

Edge_pixels=zeros(W,L);
different_pixels=zeros(W,L);
same_pixels=zeros(W,L);
image_align_gray = rgb2gray(image_align);

for i=1:W
    for j = 1:L
        if image_align_gray(i,j)==0
            Edge_pixels(i,j) = 1;
            number_EdgePixels = number_EdgePixels+1;
            image_di(i,j) = 0;
        else
            if (image_align_gray(i,j)~=0)&&(image_di(i,j)==1)
                different_pixels(i,j) = 1;
                number_DifferentPixels = number_DifferentPixels+1;
            else
                same_pixels(i,j)=1;
            end
        end
    end
end

number_AlignedPixels = sum_pixels-number_EdgePixels;
number_SamePixels = number_AlignedPixels-number_DifferentPixels;
if (number_DifferentPixels/number_AlignedPixels)>0.35
    different_pixels = same_pixels;
end
image_di = different_pixels;
end
