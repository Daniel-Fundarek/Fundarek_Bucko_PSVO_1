import cv2
import numpy as np
import glob
from ximea import xiapi


def rotate_image(image):
    height, width, channels = image.shape
    array = np.empty(shape=(width, height, channels), dtype='uint8')
    for i in range(height):
        pixel_row = image[-i]
        array[:, i] = pixel_row  # rotate pixel_row to pixel column
    return array
def openNPZ():
    data = np.load('camera_calibration.npz')
    lst = data.files
    for item in lst:
        print(item)
        print(data[item])

def capture_webcam_images(img_size, camera='ntb'):  # cam
    images = []
    i = 0
    filepath = f'circles/circle'
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
            # image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            image = detect_circle(image)
            image = cv2.resize(image, img_size)

            cv2.imshow("preview", image)
            if key == ord(' '):
                images.append(image)
                key = 0
                filepath1 = filepath + str(i) + '.png'
                cv2.imwrite(filepath1, image)
                print(f' Image {filepath1} is saved')
                i += 1

            key = cv2.waitKey(100)

        cam.stop_acquisition()
        cam.close_device()

    if camera == 'ntb':
        cam = cv2.VideoCapture(0)

        while key != ord('q'):

            ret, image = cam.read()
            image = cv2.resize(image, img_size)
            image = detect_circle(image)
            cv2.imshow("preview", image)
            if key == ord(' '):
                images.append(image)
                key = 0
                filepath1 = filepath + str(i) + '.png'

                cv2.imwrite(filepath1, image)
                print(f' Image {filepath1} is saved')
                i += 1

            key = cv2.waitKey(100)

    cv2.destroyAllWindows()
    return images


def detect_circle(img = None):
    # img = cv2.imread("circles/circle0.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("image loaded")
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 3, 1800, param1=220, param2=95, minRadius=100, maxRadius=600)
    print("sds")
    # cv2.imshow("circle", ~gray)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
    cv2.imshow("circle",img)
    return img


def camera_calibration(vert_squares, horiz_squares):
    # Defining the dimensions of checkerboard
    CHECKERBOARD = (vert_squares, horiz_squares)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []

    # Defining the world coordinates for 3D points
    obj_array = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    obj_array[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    # Extracting path of individual image stored in a given directory
    images = glob.glob('./resources/*.png')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        #ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
        #                                         cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret == True:
            print("found")
            objpoints.append(obj_array)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        cv2.imshow('img', img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    h, w = img.shape[:2]

    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    if ret:
        np.savez('camera_calibration.npz', fx=mtx[0, 0], fy=mtx[1, 1], cx=mtx[0, 2], cy=mtx[1, 2])
        print("Camera matrix : \n")
        print(mtx)
        print("dist : \n")
        print(dist)
        print("rvecs : \n")
        print(rvecs)
        print("tvecs : \n")
        print(tvecs)
