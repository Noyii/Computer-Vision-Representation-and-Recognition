function [textons] = createTextons(imStack, bank, k)
    % Given a cell array imStack of length n containing a series of n grayscale images and a filter bank bank, compute a texton “codebook” (i.e., set of quantized filter bank responses) based on a sample of filter responses from all n images. Note that a cell array can index matrices of different sizes, so each image may have a different width and height.
    % where bank is an m × m × d matrix containing d total filters, each of size m × m, and textons is a k × d matrix in which each row is a texton, i.e., one quantized filter bank response. See provided code and data below for \filterBank.mat” when applying this function, i.e., to populate bank with some common filters. Note that to reduce complexity you may randomly sample a subset of the pixels’ filter responses to be clustered. That is, not every pixel need be used.
    % Jinbin Bai <jinbin5bai@gmail.com>
    % May 2021

    % get number of images and filters
    [m, m, d] = size(bank);
    [~, n] = size(imStack);
    filteredIms = uint8.empty;
    % loop through each image
    for i = 1:n
        [h, w] = size(imStack{i});
        filteredIm = zeros(h, w, d);
        % loop through each filter and record the filter responses
        for j = 1:d
            filteredIm(:, :, j) = imfilter(imStack{i}, bank(:, :, j));
        end

        filteredIms = [filteredIms; reshape(filteredIm, h * w, d)];
    end

    % select samples randomly, one sample pixel per 2000 pixels
    [sz, ~] = size(filteredIms);
    sample = randi(sz, fix(sz / 2000), 1);
    [~, textons] = kmeans(double(filteredIms(sample, :)), k);
end
