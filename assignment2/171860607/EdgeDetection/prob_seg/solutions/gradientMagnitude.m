function [mag, theta] = gradientMagnitude(im, sigma)
    % This function should take an RGB image as input, smooth the image with Gaussian std=sigma, compute the x and y gradient values of the smoothed image, and output image maps of the gradient magnitude and orientation at each pixel. You can compute the gradient magnitude of an RGB image by taking the L2-norm of the R, G, and B gradients. The orientation can be computed from the channel corresponding to the largest gradient magnitude. The overall gradient magnitude is the L2-norm of the x and y gradients. mag and theta should be the same size as im.
    % Jinbin Bai <jinbin5bai@gmail.com>
    % May 2021
    % improve method in part c
    im=rgb2hsv(im);

    % smooth the image with Gaussian std=sigma
    filteredIm = imgaussfilt(im, sigma);
    % compute the x and y gradient values of the smoothed image
    [Gx, Gy] = gradient(filteredIm);
    % compute overall gradient magnitude which is the L2-norm of the x and y gradients
    mag = sqrt(Gx.^2 + Gy.^2);
    % compute orientation of gradient
    sintheta = Gx ./ mag;
    costheta = Gy ./ mag;
    thetas = atan(sintheta ./ costheta) + (pi .* sign(sintheta)) .* sign(-costheta);
    % compute orientation of gradient with largest magnitude over third dimension
    [Mval, Midx] = max(mag, [], 3);
    theta = zeros(size(im, 1), size(im, 2));

    for i = 1:3
        theta = theta + thetas(:, :, i) .* (Midx == i);
    end

    % compute overall gradient magnitude which is the L2-norm of the R, G, and B gradients
    mag = sqrt(sum(mag.^2, 3));
end
