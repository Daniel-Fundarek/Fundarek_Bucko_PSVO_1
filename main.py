from ximea import xiapi
import cv2


# create instance for first connected camera
cam = xiapi.Camera()



print("Open Camera")


cam.open_device()

cam.set_exposure(10000)
cam.set_param('imgdataformat','XI_RGB32')
cam.set_param('auto_wb', 1)



cam.start_acquisition()

while cv2.waitKey() != ord('q'):
