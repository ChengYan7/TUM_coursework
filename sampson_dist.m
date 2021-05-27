function sd = sampson_dist(F, x1_pixel, x2_pixel)
    % This function calculates the Sampson distance based on the fundamental matrix F

    % numerator part
    up = (x2_pixel' * F)' .* x1_pixel;
    up = sum(up,1).*sum(up,1);
    % denominator part
    e_hat = [0,-1,0;1,0,0;0,0,0];
    down_1 = e_hat*F*x1_pixel;
    down_1 = down_1.*down_1;
    down_2 = x2_pixel'*F*e_hat;
    down_2 = down_2'.*down_2';
    down = sum(down_1+down_2,1);
    % sd
    sd = up./down;
    
end