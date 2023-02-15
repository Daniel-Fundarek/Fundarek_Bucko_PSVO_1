from ximea import xiapi
import cv2


# create instance for first connected camera
#cam = xiapi.Camera()
cam=cv2.VideoCapture(0)


# print("Open Camera")
#
#
# cam.open_device()
#
# cam.set_exposure(10000)
# cam.set_param('imgdataformat','XI_RGB32')
# cam.set_param('auto_wb', 1)
# img = xiapi.Image()
# cam.start_acquisition()

for i in range(4):
    ret,image=cam.read()
    #cam.get_image(img)

    #image = img.get_image_data_numpy()
    image = cv2.resize(image,(240,240))
    cv2.imshow("test",image)
    cv2.waitKey()


'''
while cv2.waitKey() != ord('q'):
    cam.get_image(img)
    image = img.get_image_data_numpy()
    image = cv2.resize(image,(240,240))
    cv2.imshow("test",image)
    cv2.waitKey()
    cv2.imwrite(img)

'''



cam.stop_acquisition()
cam.close_device()
