import cv2
from Zadanie_3 import image_operations as img_fcn
import numpy as np
from enum import Enum
from skimage import io, color
import skimage as ski
import matplotlib.pyplot as plt
from skimage.draw import circle_perimeter
def main2():
    picture_size = (1920,1200)
    #img_fcn.camera_calibration(6, 8)
    #img_fcn.camera_calibration(6, 4)
    #img_fcn.undistor_images()
    #img = img_fcn.capture_webcam_images(picture_size,"ntb")#ntb or xiemea
    #
    # Load the image
    image = io.imread('resources/blue_circle_bigger.png')

    image = image[:, :, :3]  # remove alpha channel
    # Convert the image to grayscale
    gray_image = color.rgb2gray(image)

    # Apply Hough Transform to detect the circles
    circles = img_fcn.hough_transform_circle_second(gray_image, (20, 50), 100)

    # Draw the detected circles on the original image
    for circle in circles:
        rr, cc = circle_perimeter(circle[0], circle[1], circle[2])

        # Create masks for the coordinates that are in bounds
        y_mask = np.logical_and(rr >= 0, rr < image.shape[0])
        x_mask = np.logical_and(cc >= 0, cc < image.shape[1])
        mask = np.logical_and(x_mask, y_mask)

        # Apply the masks to the coordinate arrays
        rr = rr[mask]
        cc = cc[mask]

        # Set the circle pixels to red
        image[rr, cc] = [0, 255, 0]

    # Display the image with the detected circles
   # plt.imshow(image)
   # plt.show()
    image_orig = io.imread('resources/blue_circle.png')
    cv2.imshow("preview", image_orig)
    cv2.imshow("detected", image)
    #img_fcn.openNPZ()
    # img_fcn.detect_circle()
    cv2.waitKey()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main2()
