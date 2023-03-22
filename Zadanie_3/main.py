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



def main2():
    # Load the image and convert it to grayscale
    img = cv2.imread('circle1.jpg')
    cv2.imshow('original',img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)


    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    cv2.imshow("sobel", edges)
    cv2.waitKey(0)

    # _, edges = cv2.threshold(edges,254,255,cv2.THRESH_BINARY)
    # cv2.imshow("sobelx", abs_grad_x)
    # cv2.imshow("sobely", abs_grad_y)
    # cv2.imshow("sobel", abs_grad)
    # # cv2.imshow("sobel", abs_grad)
    #
    # cv2.waitKey(0)

    # Apply GaussianBlur to reduce noise
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Set the Hough Circle parameters
    minDist = 50
    threshold = 75#20#75
    minRadius = 25#180#20
    maxRadius = 135#400#150
    # r = 215
    accumulator = np.zeros(gray.shape, dtype=np.uint64)
    # edges = cv2.Canny(gray, 100, 200)
    # Apply the Hough Circle transform
    circles = []
    # Create circle template with radius r
    y_indexes, x_indexes = np.where(edges>180)
    # y_indexes, x_indexes = np.nonzero(edges)
    correct_circles=[]
    for r in range(minRadius,maxRadius+1):
        accumulator = np.zeros(gray.shape, dtype=np.uint8)
        print(r)
        fill_accumulator(x_indexes, y_indexes, r, accumulator)
        uint8_accumulator_normalized = accumulator
        arr_normalized = (accumulator - accumulator.min()) / (accumulator.max() - accumulator.min())  # normalize array
        uint8_accumulator_normalized = (arr_normalized * 255).astype(np.uint8)  # scale down array
        cv2.imshow('convolved', uint8_accumulator_normalized)
        cv2.waitKey(50)
        # print(np.sort(accumulator.flatten())[::-1][0:50])

        # false positive circle centre filtration
        cy,cx = np.where(accumulator >= threshold)
        value = accumulator[cy,cx] #vektor
        # (cx, cy, r, value)
        for i in range(len(cx)):
            correct_circles.append((cx[i], cy[i], r, value[i]))
        # sorted_indices = np.argsort(accumulator, axis=None)
        # cy,cx = np.unravel_index(sorted_indices,accumulator.shape)
    sorted_circles = sorted(correct_circles, key=lambda x: x[3], reverse=True)
    new_circles = [sorted_circles[0]]  # add the element with the highest value to the new list
    for i in range(1, len(sorted_circles)):
        cx1, cy1, r1, val1 = sorted_circles[i]
        add_circle = True
        for j in range(len(new_circles)):
            cx0, cy0, r0, val0 = new_circles[j]
            dist = np.sqrt((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2)
            if dist < minDist:  # set your desired threshold here
                    add_circle = False
        if add_circle:
            new_circles.append(sorted_circles[i])
    cx_list = [circle[0] for circle in new_circles]
    cy_list = [circle[1] for circle in new_circles]
    r_list = [circle[2] for circle in new_circles]
    for x,y,r in zip(cx_list,cy_list,r_list):
        # cv2.circle(accumulator,(x,y),1,(255,0,255),2)
        cv2.circle(img, (x, y), 1, (255, 0, 255), 2)
        cv2.circle(img, (x, y), r, (255, 0, 0), 2)
    # cv2.imshow('center', accumulator)
    cv2.imshow('center1', img)
    # print(idx)
    cv2.waitKey(0)






if __name__ == "__main__":
    main2()
