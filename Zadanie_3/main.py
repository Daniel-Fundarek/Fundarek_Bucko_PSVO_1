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
        accumulator = np.zeros(gray.shape, dtype=np.uint8)
        print(r)
        fill_accumulator(x_indexes, y_indexes, r, accumulator)
        # uint8_accumulator_normalized = accumulator
        # arr_normalized = (accumulator - accumulator.min()) / (accumulator.max() - accumulator.min())  # normalize array
        # uint8_accumulator_normalized = (arr_normalized * 255).astype(np.uint8)  # scale down array
        # cv2.imshow('convolved', uint8_accumulator_normalized)
        # cv2.waitKey(0)
        # print(np.sort(accumulator.flatten())[::-1][0:50])

        # #hladanei najvacsich hodnot a nasledne zobrazovanie stredov
        cy,cx = np.where(accumulator >= threshold)
        value = accumulator[cy,cx]
        # (cx, cy, r, value)
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






if __name__ == "__main__":
    main2()
