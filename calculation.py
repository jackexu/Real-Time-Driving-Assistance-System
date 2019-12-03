import numpy as np
import math
import cv2
import glob
from binarization import sobel_thresh
from birdsview import perspective
from linefit import start_fit
from time import time


def curvature_radius(binary_image, lt_fit, rt_fit, ym_per_pix = 60 / 1080):
    y_range = np.linspace(0, binary_image.shape[0], num = 50)
    lt_rad = ((1 + (2 * lt_fit[0] * np.max(y_range) + lt_fit[1]) ** 2) ** 1.5) * ym_per_pix / np.absolute(2 * lt_fit[0])
    rt_rad = ((1 + (2 * rt_fit[0] * np.max(y_range) * ym_per_pix + rt_fit[1]) ** 2) ** 1.5) * ym_per_pix / np.absolute(2 * rt_fit[0])
    if (lt_fit[0] + rt_fit[0]) <= 0:
        average_rad = -(lt_rad + rt_rad) / 2
    else:
        average_rad = (lt_rad + rt_rad) / 2
    return average_rad

def center_offset(lt_fit, rt_fit, image_height = 1080, image_width = 1920, xm_per_pix = 3.7 / 780):
    if lt_fit is not None and rt_fit is not None:
        lt_bottom = lt_fit[0] * image_height ** 2 + lt_fit[1] * image_height + lt_fit[2]
        rt_bottom = rt_fit[0] * image_height ** 2 + rt_fit[1] * image_height + rt_fit[2]
        offset = ((lt_bottom +  rt_bottom) / 2 - image_width / 2) * xm_per_pix
        offset = round(offset, 2)
    else:
        offset = '--'
    return offset

def steering_angle(average_rad, wheelbase=2.8702, steeringratio=16):
    steeringangle = np.arcsin(wheelbase / average_rad) * steeringratio
    steeringangle = math.degrees(steeringangle)
    return steeringangle


if __name__ == '__main__':
    test_images = glob.glob('test_image/*.jpg')
    for test_image in test_images:
        start_time = time()
        image = cv2.imread(test_image)
        perspective_image = perspective(image)[0]
        binary_image = sobel_thresh(perspective_image)
        lt_fit, rt_fit, out_img = start_fit(binary_image)
        average_rad = curvature_radius(binary_image, lt_fit, rt_fit)
        center = center_offset(lt_fit, rt_fit)
        steer = steering_angle(average_rad)
        process_time = time() - start_time
        print(average_rad, 'm', center, 'm', steer)
        print(process_time)
