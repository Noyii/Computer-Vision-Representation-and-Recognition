% calls the above functions appropriately using the provided images (gumballs.jpg, snake.jpg, and twins.jpg) and one other image of your choosing, and then displays the results to compare segmentations with color and texture. Include the following variants:
% • Choose parameter values (k, numRegions, winSize, etc) that yield a reasonable looking segmentations for each feature type and display results for the provided images. Of course, they won’t perfectly agree with the object boundaries. We’ll evaluate your assignment for correctness and understanding of the concepts, not the precise alignment of your regions.
% • Consider two window sizes for the texture representation, a smaller and larger one, with sizes chosen to illustrate some visible and explainable tradeoff on one of the example images.
% • Run the texture segmentation results with two different filter banks. One that uses all the provided filters, and one that uses a subset of the filters (of your choosing) so as to illustrate a visible difference in the resulting segmentations that you can explain for one of the example images. Note that the filter banks are organized by scale, orientation, and type. You might choose to make the subset you use quite limited to make the visual difference dramatic.
% Jinbin Bai <jinbin5bai@gmail.com>
% May 2021

% read the images and load the filter bank
gumballs = imread('gumballs.jpg');
snake = imread('snake.jpg');
twins = imread('twins.jpg');
planets = imread('planets.jpg');
load('filterBank.mat');

% create image stack with gray images
imStack = {rgb2gray(gumballs), rgb2gray(snake), rgb2gray(twins), rgb2gray(planets)};

% generate texton codebook
textons = createTextons(imStack, F, 10); % k=10

% gumballs
[colorLabelIm, textureLabelIm] = compareSegmentations(gumballs, F, textons, 12, 9, 3); %winSize= 9,numColorRegions= 6, numTextureRegions=3
subplot(1, 3, 1);
imshow(gumballs);
title('Original gumballs');
subplot(1, 3, 2);
imshow(label2rgb(colorLabelIm));
title('Color labeled gumballs with 9 color regions');
subplot(1, 3, 3);
imshow(label2rgb(textureLabelIm));
title('Texture labeled gumballs with 3 texture regions');

% snake
[colorLabelIm, textureLabelIm] = compareSegmentations(snake, F, textons, 5, 3, 6); %winSize=9, numColorRegions=4, numTextureRegions=5
subplot(1, 3, 1);
imshow(snake);
title('Original snake');
subplot(1, 3, 2);
imshow(label2rgb(colorLabelIm));
title('Color labeled snake with 3 color regions');
subplot(1, 3, 3);
imshow(label2rgb(textureLabelIm));
title('Texture labeled snake with 6 texture regions');

% twins
[colorLabelIm, textureLabelIm] = compareSegmentations(twins, F, textons, 15, 6, 7); % winSize=15, numColorRegions=6, numTextureRegions=7
subplot(1, 3, 1);
imshow(twins);
title('Original twins');
subplot(1, 3, 2);
imshow(label2rgb(colorLabelIm));
title('Color labeled twins with 6 color regions');
subplot(1, 3, 3);
imshow(label2rgb(textureLabelIm));
title('Texture labeled twins with 7 texture regions');

% planets
[colorLabelIm, textureLabelIm] = compareSegmentations(planets, F, textons, 9, 5, 4); % winSize=9, numColorRegions=5, numTextureRegions=4
subplot(1, 3, 1);
imshow(planets);
title('Original planets');
subplot(1, 3, 2);
imshow(label2rgb(colorLabelIm));
title('Color labeled planets with 5 color regions');
subplot(1, 3, 3);
imshow(label2rgb(textureLabelIm));
title('Texture labeled planets with 4 texture regions');

% twins different window size
[colorLabelIm1, textureLabelIm1] = compareSegmentations(twins, F, textons, 5, 6, 7);
[colorLabelIm2, textureLabelIm2] = compareSegmentations(twins, F, textons, 30, 6, 7);
subplot(1, 3, 1);
imshow(twins);
title('Original twins');
subplot(1, 3, 2);
imshow(label2rgb(textureLabelIm1));
title('Texture labeled twins with window size 5');
subplot(1, 3, 3);
imshow(label2rgb(textureLabelIm2));
title('Texture labeled twins with window size 30');

% Load different filter bank
subsetF = F(:, :, 3:5) + F(:, :, 9:11) + F(:, :, 15:17) + F(:, :, 21:23) + F(:, :, 27:29) + F(:, :, 33:35);

% Generate texton codebook
textons1 = createTextons(imStack, F, 10);
textons2 = createTextons(imStack, subsetF, 10);

[colorLabelIm1, textureLabelIm1] = compareSegmentations(twins, F, textons1, 15, 6, 7);
[colorLabelIm2, textureLabelIm2] = compareSegmentations(twins, subsetF, textons2, 15, 6, 7);
subplot(1, 3, 1);
imshow(twins);
title('Original twins');
subplot(1, 3, 2);
imshow(label2rgb(textureLabelIm1));
title('Texture labeled twins with all filters');
subplot(1, 3, 3);
imshow(label2rgb(textureLabelIm2));
title('Texture labeled twins with vertical filters');
