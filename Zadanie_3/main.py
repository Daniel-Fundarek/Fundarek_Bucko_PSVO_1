import cv2
from Zadanie_3 import image_operations as img_fcn
import numpy as np
from enum import Enum

import cv2
import numpy as np

import math

def get_circle_outline_indexes(x, y, r,array):
    indexes = []
    step = 1
    for i in range(int(x-r), int(x+r+1),step):
        for j in range(y-r, y+r+1,step):
            distance = math.sqrt((i-x)**2 + (j-y)**2)
            if math.isclose(distance, r, rel_tol=0.02):
                if 0<= i < array.shape[1] and 0<= j < array.shape[0]:
                    # indexes.append((int(i), int(j)))
                    array[j,i] += 1

    # return indexes
def main2():


    # Load the image and convert it to grayscale
    img = cv2.imread('circle.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Set the Hough Circle parameters
    dp = 1
    minDist = 50
    param1 = 50
    param2 = 30
    minRadius = 20
    maxRadius = 400
    r = 220
    accumulator = np.zeros(gray.shape, dtype=np.uint64)
    edges = cv2.Canny(gray, 100, 200)
    # Apply the Hough Circle transform
    circles = []
    # Create circle template with radius r
    test_acum = np.zeros(gray.shape, dtype=np.uint8)
    x_indexes,y_indexes = np.nonzero(edges)
    # for x in x_indexes:
    #     pass
    for x,y in zip(x_indexes, y_indexes):
        get_circle_outline_indexes(x,y,r,test_acum)

    arr_normalized = (test_acum - test_acum.min()) / (test_acum.max() - test_acum.min())  # normalize array
    arr_uint8 = (arr_normalized * 255).astype(np.uint8)  # scale down array
    cv2.imshow('convolved', arr_uint8)
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
