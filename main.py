# from ximea import xiapi
import cv2
import numpy as np

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
        item = image[-i]
        array[:, i] = item
    return array


def save_webcam_images(count):
    for i in range(count):
        ret, image = cam.read()
        # cam.get_image(img)

        # image = img.get_image_data_numpy()
        # image = cv2.resize(image, (240, 240))
        cv2.imshow("test", image)
        img_rotated = rotate_image(image)
        cv2.imshow("tdest", img_rotated)


        filepath = f'resources/img{i}.png'
        key = 0
        while key != 32:
            key = cv2.waitKey()
        cv2.imwrite(filepath, image)
        print(f' Image {filepath} is saved')


save_webcam_images(4)

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
