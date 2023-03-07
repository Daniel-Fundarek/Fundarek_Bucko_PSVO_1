import cv2
from Zadanie_2 import image_operations as img_fcn
import numpy as np
from enum import Enum

def main2():
    picture_size = 240
    img_fcn.camera_calibration(6, 8)
    img = img_fcn.capture_webcam_images("ximea")#ntb or xiemea
    cv2.waitKey()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main2()
