function [colorLabelIm, textureLabelIm] = compareSegmentations(origIm, bank, textons, winSize, numColorRegions, numTextureRegions)
    % Given an h × w × 3 RGB color image origIm, compute two segmentations: one based on color features and one based on texture features. The color segmentation should be based on k-means clustering of the colors appearing in the given image. The texture segmentation should be based on k-means clustering of the image’s texton histograms. where colorLabelIm and textureLabelIm are h×w matrices recording segment/region labels, numColorRegions and numTextureRegions specify the number of desired segments for the two feature types, and the others are defined as above.
    % Jinbin Bai <jinbin5bai@gmail.com>
    % May 2021

    % get some dimensions
    [h, w, d] = size(origIm);

    % caculate colorLabelIm
    colorLabelIm = reshape(kmeans(im2double(reshape(origIm, h * w, d)), numColorRegions), h, w);

    % caculate textureLabelIm
    textonHist = extractTextonHists(rgb2gray(origIm), bank, textons, winSize);
    textureLabelIm = reshape(kmeans(im2double(reshape(textonHist, h * w, size(textonHist, 3))), numTextureRegions), h, w);
end
