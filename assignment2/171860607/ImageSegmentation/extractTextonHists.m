function [featIm] = extractTextonHists(origIm, bank, textons, winSize)
    % Given a grayscale image, filter bank, and texton codebook, construct a texton histogram for each pixel based on the frequency of each texton within its neighborhood (as defined by a local window of fixed scale winSize). Note that textons are discrete. A pixel is mapped to one discrete texton based on its distance to each texton. (see quantizeF eats above). where textons is a k × d matrix.
    % Jinbin Bai <jinbin5bai@gmail.com>
    % May 2021

    % get some dimensions
    [h, w] = size(origIm);
    [m, m, d] = size(bank);
    [k, d] = size(textons);
    filteredIm = zeros(h, w, d);
    featIm = zeros(h, w, k);

    % loop through each filter and apply it to the image
    for i = 1:d
        filteredIm(:, :, i) = imfilter(origIm, bank(:, :, i));
    end

    % get labelIm with quantizeFeats function
    labelIm = quantizeFeats(filteredIm, textons);

    % calculate frequency of each texton in window
    for i = 1:h

        for j = 1:w
            window = labelIm(max(i - fix(winSize / 2), 1):min(i + fix(winSize / 2), h), max(j - fix(winSize / 2), 1):min(j + fix(winSize / 2), w));
            unq = unique(window);
            freq = [unq, histc(window(:), unq)];
            featIm(i, j, freq(:, 1)) = freq(:, 2);
        end

    end

end
