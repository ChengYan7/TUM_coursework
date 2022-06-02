function [ch_ratio,ch_speed] = change_analysis(number_AlignedPixels,number_DifferentPixels,num_month)
    %% change ratio
    ch_ratio = number_DifferentPixels/number_AlignedPixels ;
    
    %% change speed
    ch_speed = ch_ratio/num_month;
    
    
    
    