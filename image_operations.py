import cv2
import numpy as np

def rotate_image(image):
    height, width, channels = image.shape
    array = np.empty(shape=(width, height, channels), dtype='uint8')
    for i in range(height):
        pixel_row = image[-i]
        array[:, i] = pixel_row #rotate pixel_row to pixel column
    return array


def capture_webcam_images(count,cam):
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
