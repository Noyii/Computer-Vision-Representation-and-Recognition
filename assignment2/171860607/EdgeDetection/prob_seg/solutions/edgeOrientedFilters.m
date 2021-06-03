function bmap = edgeOrientedFilters(im)
    % Similar to part (a), this should call orientedFilterMagnitude, perform the non-maxima suppression, and output the final soft edge map.
    % Jinbin Bai <jinbin5bai@gmail.com>
    % May 2021

    
    % use gradientMagnitude to compute a soft boundary map
    [mag, theta] = orientedFilterMagnitude(im);
    % rescale the boundary scores for better visualization
    mag2 = mag.^0.7;

    % two ways to perform suppression
    % canny suppression
    edges = edge(rgb2gray(im), 'canny');
    bmap = mag2 .* edges;
    % non-max suppression
    % bmap = nonmax(mag2, theta);
    
    end