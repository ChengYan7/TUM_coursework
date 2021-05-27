function Cake = cake(min_dist)
    % The cake function creates a "cake matrix" that contains a circular set-up of zeros
    % and fills the rest of the matrix with ones. 
    % This function can be used to eliminate all potential features around a stronger feature
    % that don't meet the minimal distance to this respective feature.

    row = 2*min_dist+1; 
    column = 2*min_dist+1;
    
    r = min_dist;   
    m1 = -(row-1)/2:(row/2+1)-1;   % core of circle
    n1 = -(column-1)/2:(column/2+1)-1;
    [x,y] = meshgrid(m1,n1);
    circle = x.^2+y.^2;   % circle function

    Cake = zeros(row,column,'logical');  
    Cake(find(circle<=r*r)) = logical(0);  % contain a circular set-up of zeros
    Cake(find(circle>r*r)) = logical(1);   % fills up the rest of the matrix with ones

    
end