import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from binarization import binarize
from time import time


def perspective(image):
    height, width = image.shape[:2]
    # For test videos: real_time_driving_1 & real_time_driving_2
    src = np.float32([[908, 745],              # Top left
                      [400,  image.shape[0]],   # Bottom left
                      [1515, image.shape[0]],   # Bottom right
                      [1000, 745]])            # Top right
    # For test videos: real_time_driving_1 & real_time_driving_2
    dst = np.float32([[475,  0],                # Top left
                      [600,  1080],             # Bottom left
                      [1300, 1080],             # Bottom right
                      [1400, 0]])               # Top right
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    perspective_image = cv2.warpPerspective(image, M, (width, height), flags = cv2.INTER_LINEAR)
    return (perspective_image, M, M_inv)

def perspective_inv(image):
    height, width = image.shape[:2]
    # For test videos: real_time_driving_1 & real_time_driving_2
    src = np.float32([[908, 745],              # Top left
                      [400,  image.shape[0]],   # Bottom left
                      [1515, image.shape[0]],   # Bottom right
                      [1000, 745]])            # Top right
    # For test videos: real_time_driving_1 & real_time_driving_2
    dst = np.float32([[475,  0],                # Top left
                      [600,  1080],             # Bottom left
                      [1300, 1080],             # Bottom right
                      [1400, 0]])               # Top right
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    perspective_inv_image = cv2.warpPerspective(image, M_inv, (width, height), flags = cv2.INTER_LINEAR)
    return perspective_inv_image, M, M_inv


if __name__ == '__main__':
    start_time = time()
    image = cv2.imread('test_image/real_time_driving_1_1_4.jpg')
    binary_image = binarize(image)
    perspective_image = perspective(binary_image)[0]
    plt.imshow(perspective_image, cmap = 'gray')
    cv2.imwrite('output_image/perspective_image.png', perspective_image * 255)
    process_time = time() - start_time
    print(process_time)
    plt.show()
