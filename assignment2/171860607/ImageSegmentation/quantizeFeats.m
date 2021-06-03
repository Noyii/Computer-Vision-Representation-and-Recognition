function [labelIm] = quantizeFeats(featIm, meanFeats)
    % Given an h × w × d matrix featIm, where h and w are the height and width of the original image and d denotes the dimensionality of the feature vector already computed for each of its pixels, and given a k × d matrix meanF eats of k cluster centers, each of which is a d-dimensional vector (a row in the matrix), map each pixel in the input image to its appropriate k-means center. Return labelIm, an h × w matrix of integers indicating the cluster membership (1...k) for each pixel.
    % Jinbin Bai <jinbin5bai@gmail.com>
    % May 2021

    % get some dimensions
    [h, w, d] = size(featIm);
    [k, d] = size(meanFeats);
    labelIm = zeros(h, w);

    % loop through each column and find the cluster for that column of pixels using dist2.m
    for col = 1:w
        % create a matrix of vectors for each pixel in a column
        x = reshape(featIm(:, col, :), h, d);
        % save the cluster where the pixel is closest to in labelIm
        [~, labelIm(:, col)] = min(dist2(x, meanFeats), [], 2);
    end

end
