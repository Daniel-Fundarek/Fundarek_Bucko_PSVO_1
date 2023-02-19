# from ximea import xiapi
import cv2
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
def rotate_image(image):
    height, width, channels = image.shape
    array = np.empty(shape=(width, height, channels), dtype='uint8')
    for i in range(height):
        pixel_row = image[-i]
        array[:, i] = pixel_row #rotate pixel_row to pixel column
    return array


def capture_webcam_images(count):
    images = []
    for i in range(count):
        ret, image = cam.read()
        # cam.get_image(img)

        # image = img.get_image_data_numpy()
        # image = cv2.resize(image, (240, 240))
        images.append(image)
        cv2.imshow("test", image)



        filepath = f'resources/img{i}.png'
        key = 0
        while key != 32:
            key = cv2.waitKey()
        cv2.imwrite(filepath, image)
        print(f' Image {filepath} is saved')
    cv2.destroyAllWindows()
    return images

def select_red_channel(image):
    b, g, r = cv2.split(image)
    colored_image = cv2.merge([b*0, g*0, r])
    return colored_image
def apply_kernel_filter(mosaique):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], np.float32)  # kernel should be floating point type
    filtered_image = cv2.filter2D(mosaique, -1, kernel)
    # ddepth = -1, means destination image has depth same as input image
    return filtered_image


def main():
  img = capture_webcam_images(4)
  #array = [tuple(img[1:3]), tuple(img[0:2]), (img[2], img[1])] #num of touples == columns  #num of items in touples== rows
  img_stack=[(img[0],img[2]),(img[1],img[3])]
  mosaique = create_mosaique(img_stack)
  cv2.imshow('mosaique', mosaique)
  cv2.imwrite("resources/mosaique.png", mosaique)
  filtered_mosaique=apply_kernel_filter(mosaique)
  cv2.imshow('kernel_filtered_mosaique', filtered_mosaique)
  cv2.imwrite("resources/kernel_filtered_mosaique.png",filtered_mosaique)
  rotated_image=rotate_image(img[2])
  cv2.imshow('rotated_image', rotated_image)
  cv2.imwrite("resources/rotated_image.png", rotated_image)
  red_image=select_red_channel(img[1])
  cv2.imshow('red_image', red_image)
  cv2.imwrite("resources/red_image.png", red_image)
  cv2.waitKey()
  cv2.destroyAllWindows()


def create_mosaique(image_collection: list):
  """

  :param image_collection: List[tuple[horizontal pictures], tuple[horizontal pictures],...]
  you can add as many tuples to list as you want. Each tuple creates new vertical layer.
  Items in tuple are horizontal pictures in mosaique. you can add as many items to tuple.
  len(List) = num of vertical layers
  len(tuple) = horizontal items number

  :return: ndarray - containing concatenated all images into one according to input param
  """
  h_image_count = len(image_collection[0])
  height, width, channels = image_collection[0][0].shape
  output_array = np.zeros((h_image_count * height, 0, channels), dtype='uint8')

  for horizontal in image_collection:
      output_array = np.concatenate((output_array, np.concatenate(horizontal, axis=0)), axis=1)
  return output_array


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
