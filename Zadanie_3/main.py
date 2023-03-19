import cv2
from Zadanie_3 import image_operations as img_fcn
import numpy as np
from enum import Enum

import cv2
import numpy as np

import math


def fill_accumulator(x_indexes, y_indexes, r, accumulator):
    indexes = []
    step = 1
    thetas = np.linspace(0, 2 * np.pi, 100)
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)

    for x, y in zip(x_indexes,y_indexes):
        xs = x - r * sin_thetas
        ys = y + r * cos_thetas
        # Find the valid indices where xs_int and ys_int are within the image dimensions
        valid_indices = np.where((0 <= xs) & (xs < accumulator.shape[1]-1) & (0 <= ys) & (ys < accumulator.shape[0]-1))

        # Increment the accumulator array at the valid center coordinates
        xs_int, ys_int = np.round(xs[valid_indices]).astype(int), np.round(ys[valid_indices]).astype(int)
        accumulator[ys_int, xs_int] += 1


    # for i in range(int(x-r), int(x+r+1),step):
    #     for j in range(y-r, y+r+1,step):
    #         distance = math.sqrt((i-x)**2 + (j-y)**2)
    #         if math.isclose(distance, r, rel_tol=0.02):
    #             if 0<= i < array.shape[1] and 0<= j < array.shape[0]:
    #                 # indexes.append((int(i), int(j)))
    #                 array[j,i] += 1

    # return indexes


def main2():
    # Load the image and convert it to grayscale
    img = cv2.imread('circle1.jpg')
    cv2.imshow('original',img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Set the Hough Circle parameters
    dp = 1
    minDist = 50
    param1 = 50
    threshold = 75
    minRadius = 20
    maxRadius = 150
    # r = 215
    accumulator = np.zeros(gray.shape, dtype=np.uint64)
    edges = cv2.Canny(gray, 100, 200)
    # Apply the Hough Circle transform
    circles = []
    # Create circle template with radius r

    y_indexes, x_indexes = np.nonzero(edges)
    for r in range(minRadius,maxRadius+1):
        accumulator = np.zeros(gray.shape, dtype=np.uint16)
        print(r)
        fill_accumulator(x_indexes, y_indexes, r, accumulator)
        arr_normalized = (accumulator - accumulator.min()) / (accumulator.max() - accumulator.min())  # normalize array
        uint8_accumulator_normalized = (arr_normalized * 255).astype(np.uint8)  # scale down array
        # cv2.imshow('convolved', uint8_accumulator_normalized)
        # cv2.waitKey(0)

        print(np.sort(accumulator.flatten())[::-1][0:50])

        # #hladanei najvacsich hodnot a nasledne zobrazovanie stredov
        # k = 20
        # flatten_idx = np.argpartition(accumulator.flatten(), -k)[-k:]
        # cy,cx = idx= np.unravel_index(flatten_idx, accumulator.shape)
        cy,cx = indices = np.where(accumulator >= threshold)

        # sorted_indices = np.argsort(accumulator, axis=None)
        # cy,cx = np.unravel_index(sorted_indices,accumulator.shape)


        for x,y in zip(cx,cy):
            # cv2.circle(accumulator,(x,y),1,(255,0,255),2)
            cv2.circle(img, (x, y), 1, (255, 0, 255), 2)
            cv2.circle(img, (x, y), r, (255, 0, 0), 2)
    # cv2.imshow('center', accumulator)
    cv2.imshow('center1', img)
    # print(idx)
    cv2.waitKey(0)


    #

    # edges_ind = np.argwhere(edges>0)
    # for y,x in edges:
    #     a = x - r * np.sin(np.arcsin((y - dp * r) / r))
    #     b = y + r * np.cos(np.arcsin((y - dp * r) / r))
    #     if 0 < a < gray.shape[1] and 0 < b < gray.shape[0]:
    #         accumulator[int(b), int(a)] += 1
    #
    # for y in range(edges.shape[0]):
    #     for x in range(edges.shape[1]):
    #         if edges[y][x] > 0:
    #             pass
    #             # tu iterujeme cez kruznicu

    # Apply Hough Transform to detect the circles
    # circles = img_fcn.hough_transform_circle_second(gray_image, (20, 50), 100)


#  # Draw the detected circles on the original image
#  for circle in circles:
#      rr, cc = circle_perimeter(circle[0], circle[1], circle[2])
#
#      # Create masks for the coordinates that are in bounds
#      y_mask = np.logical_and(rr >= 0, rr < image.shape[0])
#      x_mask = np.logical_and(cc >= 0, cc < image.shape[1])
#      mask = np.logical_and(x_mask, y_mask)
#
#      # Apply the masks to the coordinate arrays
#      rr = rr[mask]
#      cc = cc[mask]
#
#      # Set the circle pixels to red
#      image[rr, cc] = [0, 255, 0]
#
#  # Display the image with the detected circles
# # plt.imshow(image)
# # plt.show()
#  image_orig = io.imread('resources/blue_circle.png')
#  cv2.imshow("preview", image_orig)
#  cv2.imshow("detected", image)
#  #img_fcn.openNPZ()
#  # img_fcn.detect_circle()
#  cv2.waitKey()
#  cv2.destroyAllWindows()


if __name__ == "__main__":
    main2()
