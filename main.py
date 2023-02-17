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

def save_webcam_images(count):
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


def main():
    img = save_webcam_images(4)
    array = [tuple(img[1:3]), tuple(img[0:2]), (img[2], img[1])]
    mosaique = create_mosaique(array)
    cv2.imshow('mosaique', mosaique)
    cv2.imwrite("mosaique.png",mosaique)
    cv2.waitKey()


def create_mosaique(vertical_array: list):
    """

    :param vertical_array: List[tuple[horizontal pictures], tuple[horizontal pictures],...]
    you can add as many tuples to list as you want. Each tuple creates new vertical layer.
    Items in tuple are horizontal pictures in mosaique. you can add as many items to tuple.
    len(List) = num of vertical layers
    len(tuple) = horizontal items number

    :return: ndarray - containing concatenated all images into one according to input param
    """
    h_image_count = len(vertical_array[0])
    height, width, channels = vertical_array[0][0].shape
    output_array = np.zeros((h_image_count * height, 0, channels), dtype='uint8')

    for horizontal in vertical_array:
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
