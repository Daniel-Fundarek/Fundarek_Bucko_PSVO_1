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


def capture_webcam_images(count, camera='ntb'):  # cam
    cam = None
    images = []

    if camera == "ximea":
        cam = xiapi.Camera()
        print("Open Camera")
        cam.open_device()
        cam.set_exposure(10000)
        cam.set_param('imgdataformat', 'XI_RGB32')
        cam.set_param('auto_wb', 1)
        img = xiapi.Image()
        cam.start_acquisition()
        cam.get_image(img)
        image = img.get_image_data_numpy()
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        image = cv2.resize(image, (240, 240))
        images.append(image)
        cv2.imshow("test", image)

        for i in range(count):
            filepath = f'resources/img{i}.png'
            key = 0
            while key != 32:
                key = cv2.waitKey()
            cv2.imwrite(filepath, image)
            print(f' Image {filepath} is saved')

        cam.stop_acquisition()
        cam.close_device()

    if camera == 'ntb':
        cam = cv2.VideoCapture(0)

        for i in range(count):
            ret, image = cam.read()

            image = cv2.resize(image, (240, 240))
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
def camera_calibration (vert_squares,horiz_squares):
    # Defining the dimensions of checkerboard
    CHECKERBOARD = (vert_squares, horiz_squares)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    # Extracting path of individual image stored in a given directory
    images = glob.glob('./images/*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret == True:
            objpoints.append(objp)
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
        np.savez('camera_calibration.npz', fx=mtx[0,0], fy=mtx[1,1],cx=mtx[0,2],cy=mtx[1,2])
        print("Camera matrix : \n")
        print(mtx)
        print("dist : \n")
        print(dist)
        print("rvecs : \n")
        print(rvecs)
        print("tvecs : \n")
        print(tvecs)