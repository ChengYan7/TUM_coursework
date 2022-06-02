function [im_gif] = gif_generation(im,date)
im_gray = {};
align_rgb = {};
len_pair = {};
im_gif = {};
threshold = 0.0008;

% image registration for further gif generation
for i = 1:numel(im)
    im_gray{i} = rgb2gray(im{i});
    if i==1
        align_rgb{1} = im{1};
        len_pair{1} = 60;
    else
        [~,align_rgb{i},~,~] = kaze(im_gray{i},im_gray{1},im{i},threshold);

    end
           im_gif{i} = insertText(align_rgb{i},[1 1],num2str(date(i),'%.2f'),'FontSize',100,...
           'BoxColor','black','BoxOpacity',0.4,'TextColor','white'); 
end