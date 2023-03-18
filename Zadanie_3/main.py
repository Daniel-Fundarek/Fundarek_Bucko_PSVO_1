import cv2
from Zadanie_2 import image_operations as img_fcn
import numpy as np
from enum import Enum

def main2():
    picture_size = (1920,1200)
    #img_fcn.camera_calibration(6, 8)
    #img_fcn.camera_calibration(6, 4)
    #img_fcn.undistor_images()
    #img = img_fcn.capture_webcam_images(picture_size,"ntb")#ntb or xiemea
    #
    img_fcn.openNPZ()
    # img_fcn.detect_circle()
    cv2.waitKey()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main2()
