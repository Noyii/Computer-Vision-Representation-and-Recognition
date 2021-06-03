function bmap = edgeGradient(im)
    % This function should use gradientMagnitude to compute a soft boundary map and then perform non-maxima suppression. For this assignment, it is acceptable to perform non-maxima suppression by retaining only the magnitudes along the binary edges produce by the Canny edge detector: edge(im,′ canny′). Alternatively, you could use the provided nonmax.m (be careful about the way orientation is defined if you do). You may obtain better results by writing a non-maxima suppression algorithm that uses your own estimates of the magnitude and orientation. If desired, the boundary scores can be rescaled, e.g., by raising to an exponent: mag2 = mag^0.7, which is primarily useful for visualization.
    % Jinbin Bai <jinbin5bai@gmail.com>
    % May 2021

    % use gradientMagnitude to compute a soft boundary map
    [mag, theta] = gradientMagnitude(im, 3);
    % rescale the boundary scores for better visualization
    mag2 = mag.^0.7;

    % two ways to perform suppression
    % canny suppression
    edges = edge(rgb2gray(im), 'canny');
    bmap = mag2 .* edges;
    % non-max suppression
    % bmap = nonmax(mag2, theta);

end
