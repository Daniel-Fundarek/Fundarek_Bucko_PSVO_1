import cv2
import numpy as np
from ximea import xiapi


def rotate_image(image):
    height, width, channels = image.shape
    array = np.empty(shape=(width, height, channels), dtype='uint8')
    for i in range(height):
        pixel_row = image[-i]
        array[:, i] = pixel_row  # rotate pixel_row to pixel column
    return array


def capture_webcam_images(camera='ntb'):  # cam
    images = []
    i = 0
    filepath = f'resources/img'
    key = 0
    if camera == "ximea":
        cam = xiapi.Camera()
        print("Open Camera")
        cam.open_device()
        cam.set_exposure(10000)
        cam.set_param('imgdataformat', 'XI_RGB32')
        cam.set_param('auto_wb', 1)
        img = xiapi.Image()
        cam.start_acquisition()

        while key != ord('q'):

            cam.get_image(img)
            image = img.get_image_data_numpy()
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            image = cv2.resize(image, (240, 240))
            cv2.imshow("preview", image)
            if key == ord(' '):
                images.append(image)
                key = 0
                filepath1 = filepath + str(i) + '.png'
                cv2.imwrite(filepath1, image)
                print(f' Image {filepath1} is saved')
                i += 1

            key = cv2.waitKey(1)

        cam.stop_acquisition()
        cam.close_device()

    if camera == 'ntb':
        cam = cv2.VideoCapture(0)

        while key != ord('q'):

            ret, image = cam.read()
            image = cv2.resize(image, (240, 240))
            cv2.imshow("preview", image)
            if key == ord(' '):
                images.append(image)
                key = 0
                filepath1 = filepath + str(i) + '.png'

                cv2.imwrite(filepath1, image)
                print(f' Image {filepath1} is saved')
                i +=1

            key = cv2.waitKey(1)

    cv2.destroyAllWindows()
    return images


def select_color_channel(image, color: str):
    b, g, r = cv2.split(image)
    # if color!="blue"or"green"or"red":
    #    return image
    if color == "blue":
        colored_image = cv2.merge([b, g * 0, r * 0])

    if color == "green":
        colored_image = cv2.merge([b * 0, g, r * 0])

    if color == "red":
        colored_image = cv2.merge([b * 0, g * 0, r])

    return colored_image


def apply_kernel_filter(mosaique, width_start, width_stop, height_start, height_stop):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], np.float32)  # kernel should be floating point type

    mosaique[width_start:width_stop, height_start:height_stop, :] = \
        cv2.filter2D(mosaique[width_start:width_stop, height_start:height_stop, :], -1, kernel)
    return mosaique


def print_img_info(image):
    print(
        f"Width is: {len(image[0, :, :])} "
        f"height is: {len(image[:, 0, :])} "
        f"and the data type is:{image.dtype}")


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
