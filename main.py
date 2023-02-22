
import cv2
import image_operations as img_fcn
import numpy as np
from enum import Enum

# create instance for first connected camera


#cam = cv2.VideoCapture(0)



def main():
    picture_size = 240
    #img = img_fcn.capture_webcam_images(4,cam)
    img = img_fcn.capture_webcam_images(4,"ximea")
    # array = [tuple(img[1:3]), tuple(img[0:2]), (img[2], img[1])] #num of touples == columns  #num of items in touples== rows
    img_stack = [(img[0], img[2]), (img[1], img[3])]
    mosaique = img_fcn.create_mosaique(img_stack)

    cv2.imshow('mosaique', mosaique)
    cv2.imwrite("resources/mosaique.png", mosaique)

    mosaique[0:picture_size, 0:picture_size, :] = img_fcn.apply_kernel_filter(
        mosaique[0:picture_size, 0:picture_size, :], 100, 200, 100, 200)

    mosaique[picture_size:picture_size * 2, 0:picture_size, :] = img_fcn.rotate_image(
        mosaique[picture_size:picture_size * 2, 0:picture_size, :])

    mosaique[0:picture_size, picture_size:picture_size * 2, :] = img_fcn.select_red_channel(
        mosaique[0:picture_size, picture_size:picture_size * 2, :], "red")

    cv2.imshow('mosaique_post_processing', mosaique)
    cv2.imwrite("resources/mosaique.png", mosaique)

    img_fcn.print_img_info(mosaique)
    cv2.waitKey()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

'''
while cv2.waitKey() != ord('q'):
cam.get_image(img)
image = img.get_image_data_numpy()
image = cv2.resize(image,(240,240))
cv2.imshow("test",image)
cv2.waitKey()
cv2.imwrite(img)

'''

# cam.stop_acquisition()
# cam.close_device()
