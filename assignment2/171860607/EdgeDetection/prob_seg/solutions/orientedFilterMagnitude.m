function [mag, theta] = orientedFilterMagnitude(im)
    % Computes the boundary magnitude and orientation using a set of oriented filters, such as elongated Gaussian derivative filters. Explain your choice of filters. Use at least four orientations. One way to combine filter responses is to compute a boundary score for each filter (simply by filtering with it) and then use the max and argmax over filter responses to compute the magnitude and orientation for each pixel.
    % Jinbin Bai <jinbin5bai@gmail.com>
    % May 2021

    % improve method in part c
    im=rgb2hsv(im);

    % convert 3 channel image to 1 channel and convert to double for increased precision
    % if(ndims(im) == 3)
    %     im = double(rgb2gray(im));
    % end

    sigma = 3;

    % find two derived gaussians with respect to x and y
    [Fx, Fy] = gradient(fspecial('gaussian', 20, sigma));

    % create another three pair filters, m means minus
    F45 = Fx * cosd(45) + Fy * sind(45);
    Fm45 = Fx * cosd(-45) + Fy * sind(-45);

    F30 = Fx * cosd(30) + Fy * sind(30);
    Fm60 = Fx * cosd(-60) + Fy * sind(-60);

    F60 = Fx * cosd(60) + Fy * sind(60);
    Fm30 = Fx * cosd(-30) + Fy * sind(-30);

    % [Fxx, Fxy] = gradient(Fx);
    % [Fyx, Fyy] = gradient(Fy);

    % run four differnet elongated Gaussian derivative filters on image
    Gx1 = double(imfilter(im, Fx, 'replicate', 'conv'));
    Gy1 = double(imfilter(im, Fy, 'replicate', 'conv'));
    Gx2 = double(imfilter(im, F45, 'replicate', 'conv'));
    Gy2 = double(imfilter(im, Fm45, 'replicate', 'conv'));
    Gx3 = double(imfilter(im, F30, 'replicate', 'conv'));
    Gy3 = double(imfilter(im, Fm60, 'replicate', 'conv'));
    Gx4 = double(imfilter(im, F60, 'replicate', 'conv'));
    Gy4 = double(imfilter(im, Fm30, 'replicate', 'conv'));
    
    % works not good
    % Gx3 = double(imfilter(im,Fxx,'replicate', 'conv'));
    % Gy3 = double(imfilter(im,Fxy,'replicate', 'conv'));
    % Gx4 = double(imfilter(im,Fyx,'replicate', 'conv'));
    % Gy4 = double(imfilter(im,Fyy,'replicate', 'conv'));

    % way 1
    % directly compute mag, as a result there is no way to compute theta
    mag=sqrt(Gx1.^2 + Gy1.^2)+sqrt(Gx2.^2 + Gy2.^2)+sqrt(Gx3.^2 + Gy3.^2)+sqrt(Gx4.^2 + Gy4.^2);

    [mag, ] = max(sqrt(mag), [], 3);
    theta = zeros(size(im, 1), size(im, 2));
    
    % % way 2 
    % % filter then use max function to vote, in this way we can compute theta
    % %%%
    % % compute overall gradient magnitude which is the L2-norm of the x and y gradients
    % mag1 = sqrt(Gx1.^2 + Gy1.^2);
    % % compute orientation of gradient
    % sintheta1 = Gx1 ./ mag1;
    % costheta1 = Gy1 ./ mag1;
    % thetas1 = atan(sintheta1 ./ costheta1) + (pi .* sign(sintheta1)) .* sign(-costheta1);
    % % compute orientation of gradient with largest magnitude over third dimension
    % [Mval1, Midx1] = max(mag1, [], 3);
    % theta1 = zeros(size(im, 1), size(im, 2));

    % for i = 1:3
    %     theta1 = theta1 + thetas1(:, :, i) .* (Midx1 == i);
    % end

    % % compute overall gradient magnitude which is the L2-norm of the R, G, and B gradients
    % mag1 = sqrt(sum(mag1.^2, 3));

    % %%%
    % % compute overall gradient magnitude which is the L2-norm of the x and y gradients
    % mag2 = sqrt(Gx2.^2 + Gy2.^2);
    % % compute orientation of gradient
    % sintheta2 = Gx2 ./ mag2;
    % costheta2 = Gy2 ./ mag2;
    % thetas2 = atan(sintheta2 ./ costheta2) + (pi .* sign(sintheta2)) .* sign(-costheta2) + pi / 4;
    % % compute orientation of gradient with largest magnitude over third dimension
    % [Mval2, Midx2] = max(mag2, [], 3);
    % theta2 = zeros(size(im, 1), size(im, 2));

    % for i = 1:3
    %     theta2 = theta2 + thetas2(:, :, i) .* (Midx2 == i);
    % end

    % %%%
    % % compute overall gradient magnitude which is the L2-norm of the R, G, and B gradients
    % mag2 = sqrt(sum(mag2.^2, 3));

    % % compute overall gradient magnitude which is the L2-norm of the x and y gradients
    % mag3 = sqrt(Gx3.^2 + Gy3.^2);
    % % compute orientation of gradient
    % sintheta3 = Gx3 ./ mag3;
    % costheta3 = Gy3 ./ mag3;
    % thetas3 = atan(sintheta3 ./ costheta3) + (pi .* sign(sintheta3)) .* sign(-costheta3) + pi / 6;
    % % compute orientation of gradient with largest magnitude over third dimension
    % [Mval3, Midx3] = max(mag3, [], 3);
    % theta3 = zeros(size(im, 1), size(im, 2));

    % for i = 1:3
    %     theta3 = theta3 + thetas3(:, :, i) .* (Midx3 == i);
    % end

    % %%%
    % % compute overall gradient magnitude which is the L2-norm of the R, G, and B gradients
    % mag3 = sqrt(sum(mag3.^2, 3));

    % % compute overall gradient magnitude which is the L2-norm of the x and y gradients
    % mag4 = sqrt(Gx4.^2 + Gy4.^2);
    % % compute orientation of gradient
    % sintheta4 = Gx4 ./ mag4;
    % costheta4 = Gy4 ./ mag4;
    % thetas4 = atan(sintheta4 ./ costheta4) + (pi .* sign(sintheta4)) .* sign(-costheta4) + pi / 3;
    % % compute orientation of gradient with largest magnitude over third dimension
    % [Mval4, Midx4] = max(mag4, [], 3);
    % theta4 = zeros(size(im, 1), size(im, 2));

    % for i = 1:3
    %     theta4 = theta4 + thetas4(:, :, i) .* (Midx4 == i);
    % end

    % % compute overall gradient magnitude which is the L2-norm of the R, G, and B gradients
    % mag4 = sqrt(sum(mag4.^2, 3));

    % % merge all four filters
    % mags(:, :, 1) = mag1; mags(:, :, 2) = mag2; mags(:, :, 3) = mag3; mag(:, :, 4) = mag4;
    % thetas(:, :, 1) = theta1; thetas(:, :, 2) = theta2; thetas(:, :, 3) = theta3; thetas(:, :, 4) = theta4;

    % [mag, Midx] = max(mags, [], 3);
    % theta = zeros(size(im, 1), size(im, 2));

    % for i = 1:4
    %     theta = theta + thetas(:, :, i) .* (Midx == i);
    % end

end
