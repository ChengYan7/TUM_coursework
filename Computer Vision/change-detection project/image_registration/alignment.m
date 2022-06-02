function [align_gray,align_rgb,fixed,moving] = alignment(tar_image,ref_image,threshold)

% image alignment
[align_gray,align_rgb,fixed,moving] = kaze(rgb2gray(tar_image),rgb2gray(ref_image),tar_image,threshold);


end
