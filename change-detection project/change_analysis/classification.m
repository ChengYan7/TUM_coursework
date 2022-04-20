function [highlight_nature,highlight_urban,isUrban]=classification(image_align)
    %% Initialization
    image_hsv = rgb2hsv(image_align);
    [m,n,~] = size(image_hsv);
    sum_pixels = m*n;
    
    %% Set threshold for dividing green or blue areas
    g = zeros(m,n);
    b = zeros(m,n);
    w = zeros(m,n);
    
    for i=1:m
        for j=1:n
            if(image_hsv(i,j,1)>35/180)&&(image_hsv(i,j,1)<90/180)&&(image_hsv(i,j,3)>46/255)&&(image_hsv(i,j,3)<221/255)
                g(i,j)=1;
            end
            if(image_hsv(i,j,1)>100/180)&&(image_hsv(i,j,1)<125/180)&&(image_hsv(i,j,3)>46/255)&&(image_hsv(i,j,3)<221/255)
                b(i,j)=1;
            end
            if (image_hsv(i,j,2)<30/255)&&(image_hsv(i,j,3)>211/255)
                w(i,j)=1;
            end
        end
    end
    nature_ratio=(sum(g,'all')+sum(b,'all')+sum(w,'all'))/sum_pixels;
    
    %% location classification
    % straight line detection
    % Reference: From external resourse
    lines = line_detector(image_align);
    lineCount = size(lines,2);
    for lineIndex=1:lineCount
        x1 = lines(2,lineIndex);
        x2 = lines(4,lineIndex);
        y1 = lines(1,lineIndex);
        y2 = lines(3,lineIndex);
        highlight_urban = insertShape(image_align,'Line',[x1 y1 x2 y2],'LineWidth',2,'Color','red');
    end
    
    if lineCount >=130 && nature_ratio<0.7
        isUrban = true;
        highlight_nature=image_align+uint8(cat(3,g*0*255,g*0.5*255,g*0*255));
    else 
        isUrban = false;
        highlight_nature=image_align+uint8(cat(3,g*0*255,g*0.5*255,g*0*255))+uint8(cat(3,b*0*255,b*0*255,b*0.5*255))+uint8(cat(3,w*1*255,w*1*255,w*1*255));
    end
end