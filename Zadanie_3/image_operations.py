import cv2
import numpy as np
import glob
from ximea import xiapi
import numpy as np
from skimage import io, color
import skimage as ski
def hough_transform_circle_second(img, radius_range, min_votes):
    rows, cols = img.shape
    rmin, rmax = radius_range
    rrange = rmax - rmin

    # Initialize accumulator array
    accumulator = np.zeros((rows, cols, rrange))

    # Precompute sin and cos values
    thetas = np.linspace(0, 2 * np.pi, 100)
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)

    # Loop over each radius to search for
    for r_idx, r in enumerate(range(rmin, rmax)):
        print(f'Processing radius: {r}')

        # Create circle template with radius r
        circle = np.zeros((2 * r + 1, 2 * r + 1))
        y, x = np.ogrid[-r:r+1, -r:r+1]
        mask = x**2 + y**2 <= r**2
        circle[mask] = 1

        # Convolve image with circle template to count edge pixels
        convolved = cv2.filter2D(img, -1, circle)

        # Loop over each edge pixel in the image
        edge_pixels = np.argwhere(convolved > 0)
        for y, x in edge_pixels:

            # Calculate the possible circle center coordinates for this pixel and radius
            xs = x - r * sin_thetas
            ys = y + r * cos_thetas
            # Find the valid indices where xs_int and ys_int are within the image dimensions
            valid_indices = np.where((xs >= 0) & (xs < cols-1) & (ys >= 0) & (ys < rows-1))

            # Increment the accumulator array at the valid center coordinates
            xs_int, ys_int = np.round(xs[valid_indices]).astype(int), np.round(ys[valid_indices]).astype(int)
            accumulator[ys_int, xs_int, r_idx] += 1

    # Find circles with the highest number of votes
    max_votes = np.max(accumulator, axis=2)
    max_coords = np.argwhere(max_votes > min_votes)
    max_radii = rmin + np.argmax(accumulator, axis=2)[max_coords[:, 0], max_coords[:, 1]]
    circles = np.concatenate((max_coords, np.reshape(max_radii, (-1, 1))), axis=1)

    return circles

def hough_transform_circle(img, radius_min, radius_max, threshold):
    """
    Detect circles in an image using the Hough Transform.

    Parameters:
    img (ndarray): The input image as a grayscale or color image.
    radius_min (int): The minimum radius to search for.
    radius_max (int): The maximum radius to search for.
    threshold (int): The minimum number of votes required for a circle to be detected.

    Returns:
    A list of tuples (x, y, r) representing the detected circles, where (x, y) is the center and r is the radius.
    """

    # Convert the input image to grayscale
    if len(img.shape) > 2:
        img = ski.rgb2gray(img)

    # Create an accumulator array with dimensions (num_rows, num_cols, num_radii)
    rows, cols = img.shape
    radii = np.arange(radius_min, radius_max+1)
    accumulator = np.zeros((rows, cols, radii.size), dtype=np.uint16)

    # Define the range of theta values (in radians) to search for
    thetas = np.deg2rad(np.arange(0, 360))

    # Loop over each edge pixel in the image
    edge_pixels = np.argwhere(img > 0)
    for y, x in edge_pixels:

        # Loop over each radius to search for
        for r_idx, r in enumerate(radii):

            # Calculate the possible circle center coordinates for this pixel and radius
            xs = x - r * np.sin(thetas)
            ys = y + r * np.cos(thetas)
            # Find the valid indices where xs_int and ys_int are within the image dimensions
            valid_indices = np.where((xs >= 0) & (xs < cols-1) & (ys >= 0) & (ys < rows-1))

            # Increment the accumulator array at the valid center coordinates
            xs_int, ys_int = np.round(xs[valid_indices]).astype(int), np.round(ys[valid_indices]).astype(int)
            accumulator[ys_int, xs_int, r_idx] += 1
            # Increment the accumulator array at the possible center coordinates
        print(f"row: {y} column: {x}")
    # Find the (x, y, r) coordinates of the circle centers with the most votes
    candidates = np.argwhere(accumulator >= threshold)
    circle_coords = [(x, y, radii[r_idx]) for y, x, r_idx in candidates]

    return circle_coords
