# In this exercise, I will implement an image stitcher that uses image warping and homographies to automati- cally create an image mosaic. We will focus on the case where we have two input images that should form the mosaic, where we warp one image into the plane of the second image and display the combined views. This problem will give some practice manipulating homogeneous coordinates, computing homography matrices, and performing image warps.
# Created by Jinbin Bai
# jinbin5bai@gmail.com

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import RectBivariateSpline
import math

# Getting correspondences: first get manually identified corresponding points from two views. Look at Matlab’s ginput function for an easy way to collect mouse click positions. The results will be sensitive to the accuracy of the corresponding points; when providing clicks, choose distinctive points in the image that appear in both views.
def getting_correspondences(A, B, numPoints=8):
    imgA = Image.open(A)
    imgB = Image.open(B)
    print("Attention you need to click A and B in turn for several times. ")
    fig = plt.figure()
    figA = fig.add_subplot(1, 2, 1)
    plt.title("Please click {num} points".format(num=int(numPoints / 2)))
    figB = fig.add_subplot(1, 2, 2)
    plt.title("Please click {num} points".format(num=int(numPoints/2)))
    figA.imshow(imgA, origin='upper')
    figB.imshow(imgB, origin='upper')

    points = plt.ginput(numPoints, timeout=0)
    # in the order of ABABABAB...
    points = np.reshape(points, (int(numPoints / 2), -1))
    ptA = points[:, [0, 1]]
    ptB = points[:, [2, 3]]
    return ptA, ptB

# Computing the homography parameters
# reference theory 3.1.1: https://www.pythonf.cn/read/93892
def computing_homography_parameters(ptA, ptB, quick_compute=False):

    if quick_compute:
        # simple way
        numPoints = len(ptA)*2
        A = np.zeros((numPoints, 8))
        for i in range(int(numPoints/2)):
            A[2*i] = [ptA[i][0], ptA[i][1], 1, 0, 0, 0, -
                      ptA[i][0]*ptB[i][0], -ptA[i][1]*ptB[i][0]]
            A[2*i+1] = [0, 0, 0, ptA[i][0], ptA[i][1], 1, -
                        ptA[i][0]*ptB[i][1], -ptA[i][1]*ptB[i][1]]

        B = np.reshape(ptB, (numPoints, 1))
        H_ = np.linalg.lstsq(A, B, rcond=None)[0]
        H = np.insert(H_, [8], [1.0]).reshape(3, 3)
        return H
    else:
        # add normalization
        fp = np.insert(ptA, 2, 1, axis=1).transpose()
        tp = np.insert(ptB, 2, 1, axis=1).transpose()

        m = np.mean(fp[:2], axis=1)
        maxstd = max(np.std(fp[:2], axis=1)) + 1e-9
        C1 = np.diag([1 / maxstd, 1 / maxstd, 1])
        C1[0][2] = -m[0] / maxstd
        C1[1][2] = -m[1] / maxstd
        origin = np.dot(C1, fp)

        m = np.mean(tp[:2], axis=1)
        maxstd = max(np.std(tp[:2], axis=1)) + 1e-9
        C2 = np.diag([1 / maxstd, 1 / maxstd, 1])
        C2[0][2] = -m[0] / maxstd
        C2[1][2] = -m[1] / maxstd
        dest = np.dot(C2, tp)

        nbr_correspondences = origin.shape[1]
        a = np.zeros((2 * nbr_correspondences, 9))
        for i in range(nbr_correspondences):
            a[2 * i] = [-origin[0][i], -origin[1][i], -1, 0, 0, 0, dest[0][i] * origin[0][i], dest[0][i] * origin[1][i],
                        dest[0][i]]
            a[2 * i + 1] = [0, 0, 0, -origin[0][i], -origin[1][i], -1, dest[1][i] * origin[0][i],
                            dest[1][i] * origin[1][i],
                            dest[1][i]]
        u, s, v = np.linalg.svd(a)
        H = v[8].reshape((3, 3))
        H = np.dot(np.linalg.inv(C2), np.dot(H, C1))
        H = H / H[2, 2]
        return H


# Verify that the homography matrix your function computes is correct by mapping the clicked image points from one view to the other, and displaying them on top of each respective image (imshow, followed by hold on and plot). Be sure to handle homogenous and non-homogenous coordinates correctly.
def verify_homography_matrix(imgA, imgB, H):
    imgA = cv2.imread(imgA)
    imgB = cv2.imread(imgB)

    fig = plt.figure()
    figA = fig.add_subplot(1, 2, 1)
    figB = fig.add_subplot(1, 2, 2)
    figA.imshow(imgA, origin='upper')
    figB.imshow(imgB, origin='upper')

    print("Click the left image and see the points showing in two images, e.g. Click the tree.")
    pts = plt.ginput(1)
    while True:
        pts = plt.ginput(1)
        print(pts)

        pts = np.reshape(pts, (1 * 2, -1))
        toTrans = np.array([pts[0], pts[1], 1], dtype=object).transpose()
        p = np.dot(H, toTrans).transpose()
        x = p[0] / p[2]
        y = p[1] / p[2]
        figA.scatter(pts[0], pts[1])
        figB.scatter([x], [y])


# Warping between image planes [25 points]: write a function that can take the recovered homography matrix and an image, and return a new image that is the warp of the input image using H . Since the transformed coordinates will typically be sub-pixel values, you will need to sample the pixel values from nearby pixels. For color images, warp each RGB channel separately and then stack together to form the output.
# To avoid holes in the output, use an inverse warp. Warp the points from the source image into the reference frame of the destination, and compute the bounding box in that new reference frame. Then sample all points in that destination bounding box from the proper coordinates in the source image. Note that transforming all the points will generate an image of a different shape / dimensions than the original input.
def warping_between_image_planes(A, B, H):
    imgA = Image.open(A)
    imgB = Image.open(B)
    w, h = imgA.size

    # PART 1
    # obtain the mapping relationship of each point coordinate
    img1_x = np.arange(0, w, 1)
    img1_y = np.arange(0, h, 1)
    X, Y = np.meshgrid(img1_x, img1_y)
    new_corr = []
    minX = minY = maxX = maxY = 0
    for i, col in enumerate(X):
        row = Y[i]
        p = np.vstack((col, row)).transpose()
        # transform A(p) to B(p_)
        p1 = np.insert(p, 2, 1, axis=1).transpose()
        tmpMatrix = H.dot(p1).transpose()
        p_ = []
        for tmpM in tmpMatrix:
            p_.append([tmpM[0] / tmpM[2], tmpM[1] / tmpM[2]])
        p_ = np.array(p_)

        xMax, yMax = p_.max(axis=0)
        xMin, yMin = p_.min(axis=0)

        minX = min(xMin, minX)
        minY = min(yMin, minY)
        maxX = max(xMax, maxX)
        maxY = max(yMax, maxY)
        new_corr.append(p_)

    # get the mapped picture size
    warpping_h = int(maxY-minY+2)
    warpping_w = int(maxX-minX+2)
    result = np.zeros((warpping_h, warpping_w, 3), dtype=np.uint8)
    pix = np.reshape(list(imgA.getdata()), (h, w, 3))

    # mapping
    for i, row in enumerate(new_corr):
        for j, corr in enumerate(row):
            x = int(corr[0]-minX+0.5)
            y = int(corr[1]-minY+0.5)
            result[y][x] = pix[i][j]

    warpping_img = Image.fromarray(result)
    warpping_img.show()

    # PART 2
    # transformed picture after inverse warpping
    x_i = np.arange(minX, maxX, 1)
    y_i = np.arange(minY, maxY, 1)
    X_i, Y_i = np.meshgrid(x_i, y_i)
    new_corr_i = []
    maxX_i = maxY_i = minX_i = minY_i = 0
    for i, col in enumerate(X_i):
        row = Y_i[i]
        p_i = np.vstack((col, row)).T

        # transform B(p_) to A(p)
        p1i = np.insert(p_i, 2, 1, axis=1).transpose()
        tmpMatrix_i = np.linalg.inv(H).dot(p1i).transpose()
        pi = []
        for tmpM in tmpMatrix_i:
            pi.append([tmpM[0] / tmpM[2], tmpM[1] / tmpM[2]])
        pi = np.array(pi)

        xMax_i, yMax_i = pi.max(axis=0)
        xMin_i, yMin_i = pi.min(axis=0)

        # print(xmin,ymin,xmax,ymax)
        minX_i = min(xMin_i, minX_i)
        minY_i = min(yMin_i, minY_i)
        maxX_i = max(xMax_i, maxX_i)
        maxY_i = max(yMax_i, maxY_i)
        new_corr_i.append(pi)

    height_i = maxY_i - minY_i
    width_i = maxX_i - minX_i
    warpping_x = np.arange(0, width_i, 1)
    warpping_y = np.arange(0, height_i, 1)

    channels = []
    for n in range(3):
        channel = pix[:, :, n]
        interp_spline = RectBivariateSpline(img1_y, img1_x, channel)
        channels.append(interp_spline(warpping_y, warpping_x))

    for i, row in enumerate(new_corr_i):
        for j, corr in enumerate(row):
            x = int(corr[0])
            y = int(corr[1])
            if x >= 0 and y >= 0:
                result[i][j] = [channels[0][y][x],
                                channels[1][y][x], channels[2][y][x]]

    reverse_warpping_img = Image.fromarray(result)
    reverse_warpping_img.show()

    # PART 3
    # merge image
    width, height = imgB.size
    data = np.zeros((int(max(height, maxY) - minY + 2), int(max(width, maxX) - minX + 2), 3),
                    dtype=np.uint8)

    # print left image
    for y in range(0, warpping_h):
        for x in range(0, warpping_w):
            if result[y][x].any() != 0:
                data[y][x] = result[y][x]

    # print right image
    pix = list(imgB.getdata())
    pix = np.reshape(pix, (height, width, 3))
    for x in range(0, width):
        for y in range(0, height):
            data[int(y - minY), int(x - minX)] = pix[y][x]

    final_img = Image.fromarray(data)
    final_img.show()


def getting_correspondences_sift(img1, img2):

    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    # down sample if the imge size is so big
    # img1 = cv2.pyrDown(img1)
    # img2= cv2.pyrDown(img2)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(400)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Initialize lists
    list_kp1 = []
    list_kp2 = []
    good_matches = []

    for mat in matches:
        # Apply ratio test
        if mat[0].distance < 0.7 * mat[1].distance:
            # Append good matches
            good_matches.append([mat[0]])
            # Get the matching keypoints for each of the images
            img1_idx = mat[0].queryIdx
            img2_idx = mat[0].trainIdx

            # Get the coordinates
            (x1, y1) = kp1[img1_idx].pt
            (x2, y2) = kp2[img2_idx].pt

            # Append to each list
            list_kp1.append((x1, y1))
            list_kp2.append((x2, y2))

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(gray1, kp1, gray2, kp2,
                              good_matches, flags=2, outImg=None)
    plt.imshow(img3), plt.show()

    return list_kp1, list_kp2


if __name__ == '__main__':
    imgA = 'uttower2.jpg'
    imgB = 'uttower1.jpg'

    # down sample if the imge size is so big
    # imgA='srcA1.jpg'
    # imgB='srcB1.jpg'
    #
    # img1 = cv2.imread(imgA)
    # img2 = cv2.imread(imgB)
    #
    #
    #
    # img1 = cv2.pyrDown(img1)
    # img2= cv2.pyrDown(img2)
    # img1 = cv2.pyrDown(img1)
    # img2 = cv2.pyrDown(img2)
    # img1 = cv2.pyrDown(img1)
    # img2 = cv2.pyrDown(img2)
    #
    # cv2.imwrite('srcA1.jpg',img1)
    # cv2.imwrite('srcB1.jpg',img2)

    mode = input(
        'Choosing point manually? default is False. Type True or Enter')
    H1 = np.zeros((3, 3))
    H2 = np.zeros((3, 3))
    if mode:
        ptA, ptB = getting_correspondences(imgA, imgB, 10)
        H1 = computing_homography_parameters(ptA, ptB)
    else:
        ptA, ptB = getting_correspondences_sift(imgA, imgB)
        H2, status = cv2.findHomography(np.asarray(ptA), np.asarray(ptB), 0, 5.0)  # without using RANSAC
        H2, status = cv2.findHomography(np.asarray(ptA), np.asarray(
            ptB), cv2.RANSAC, 5.0)  # https://zhuanlan.zhihu.com/p/36301702

    # print("H1:",H1)
    # print("H2",H2)
    # print("difference between the two matricies")
    # print(np.subtract(H1,H2))
    # print("total error = ",sum(sum(abs(np.subtract(H1,H2))))//9)

    # verify_homography_matrix(imgA,imgB,H2)
    # warping_between_image_planes(imgA,imgB,H1)
    warping_between_image_planes(imgA, imgB, H2)
