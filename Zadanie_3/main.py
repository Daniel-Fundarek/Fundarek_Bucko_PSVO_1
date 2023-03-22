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


# Define the kernel

def myConvolution(img, kernel):
    # Get the dimensions of the image and the kernel
    img_height, img_width = img.shape
    kernel_height, kernel_width = kernel.shape

    # Define an empty output image
    output = np.zeros((img_height - kernel_height + 1, img_width - kernel_width + 1))

    # Loop over each pixel in the output image
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            # Get the current patch of the image
            patch = img[i:i + kernel_height, j:j + kernel_width]

            # Perform the convolution operation
            output[i, j] = np.sum(patch * kernel)

    # Normalize the output image to [0, 255] range
    #output = (output - np.min(output)) * 255 / (np.max(output) - np.min(output))

    # Convert the output image to uint8 format
    #output = output.astype(np.uint8)
    return output


def myAddWeighted(src1, alpha, src2, beta, gamma):
    # Normalize the input arrays to [0, 1] range
    src1_norm = src1.astype(np.float32) / 255.0
    src2_norm = src2.astype(np.float32) / 255.0

    # Perform the weighted addition
    result_norm = alpha * src1_norm + beta * src2_norm + gamma

    # Convert the result back to [0, 255] range
    result = np.clip(result_norm * 255, 0, 255).astype(np.uint8)

    return result


def myConvertScaleAbs(src, alpha=1.0, beta=0.0):
    # Scale the input array by alpha
    scaled = src.astype(np.float32) * alpha

    # Add the beta offset
    scaled += beta

    # Ensure that the values are within the [0, 255] range
    scaled = np.clip(scaled, 0, 255)

    # Convert the scaled array to the absolute representation
    dst = scaled.round().astype(np.uint8)

    return dst
def main2():
    # Load the image and convert it to grayscale
    img = cv2.imread('circle1.jpg')
    cv2.imshow('original',img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    kernel_y = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]])
    kernel_x = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]])
    grad_x = myConvolution(gray,kernel_x)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = myConvolution(gray,kernel_y)


    abs_grad_x = myConvertScaleAbs(grad_x)
    abs_grad_y = myConvertScaleAbs(grad_y)
    edges = myAddWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
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
    minDist = 150
    threshold = 40#20#75
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
