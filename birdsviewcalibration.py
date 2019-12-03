import numpy as np
import cv2
import matplotlib.pyplot as plt

def draw_ref_points(image):
    print(image.shape)
    # For test videos: real_time_driving_1 & real_time_driving_2
    # Top left
    cv2.circle(image, (908, 745), 5, (255, 0, 0), -1)
    # Bottom left
    cv2.circle(image, (400, image.shape[0]), 5, (255, 0, 0), -1)
    # Bottom right
    cv2.circle(image, (1515, image.shape[0]), 5, (255, 0, 0), -1)
    # Top right
    cv2.circle(image, (1000, 745), 5, (255, 0, 0), -1)
    return image


def birdsview_calibration(image):
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
    return perspective_image, M, M_inv


if __name__ == '__main__':
    image = cv2.imread('test_image/real_time_driving_1_1_1.jpg')
    points_image = draw_ref_points(image)
    points_image = cv2.resize(points_image, (1440, 810))
    birdsview_image, _, _ = birdsview_calibration(image)
    birdsview_image = cv2.resize(birdsview_image, (1440, 810))
    plt.imshow(points_image)
    plt.figure()
    plt.imshow(birdsview_image)
    plt.show()
