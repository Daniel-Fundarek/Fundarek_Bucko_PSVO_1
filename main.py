# from ximea import xiapi
import cv2
import image_operations as img_fcn
import numpy as np
from enum import Enum
# create instance for first connected camera
# cam = xiapi.Camera()

cam = cv2.VideoCapture(0)


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

def main():
  img = img_fcn.capture_webcam_images(4,cam)
  #array = [tuple(img[1:3]), tuple(img[0:2]), (img[2], img[1])] #num of touples == columns  #num of items in touples== rows
  img_stack=[(img[0],img[2]),(img[1],img[3])]
  mosaique = img_fcn.create_mosaique(img_stack)
  cv2.imshow('mosaique', mosaique)
  cv2.imwrite("resources/mosaique.png", mosaique)
  filtered_mosaique=img_fcn.apply_kernel_filter(mosaique[0:240,0:240,:],100,200,100,200)
  cv2.imshow('kernel_filtered_mosaique', filtered_mosaique)
  cv2.imwrite("resources/kernel_filtered_mosaique.png",filtered_mosaique)
  rotated_image=img_fcn.rotate_image(img[2])
  cv2.imshow('rotated_image', rotated_image)
  cv2.imwrite("resources/rotated_image.png", rotated_image)
  red_image=img_fcn.select_red_channel(img[1],"red")
  cv2.imshow('red_image', red_image)
  cv2.imwrite("resources/red_image.png", red_image)
  img_fcn.print_img_info(red_image)
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
