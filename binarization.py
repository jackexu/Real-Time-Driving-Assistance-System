import numpy as np
import cv2
import matplotlib.pyplot as plt
from time import time


def region_of_interest_edge(image):
    height, width = image.shape[:2]
    triangle = np.array([
        [(0 + 300, height), (width - 300, height), (int(width - 700), 800), (int(0 + 700), 800)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    # masked_image = cv2.addWeighted(image, 1, mask, 0.2, 0)
    return masked_image


def region_of_interest_color(image):
    height, width = image.shape[:2]
    triangle = np.array([
        [(0 + 300, height), (width - 300, height), (int(width * 0.5), int(height * 0.6))]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    # masked_image = cv2.addWeighted(image, 1, mask, 0.2, 0)
    return masked_image


def canny_edge(image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2GRAY)
    binary_output = cv2.Canny(gray_image, 50, 150)
    binary_output[binary_output < 255] = 0
    binary_output[binary_output == 255] = 1
    binary_output = region_of_interest_edge(binary_output)
    return binary_output


def sobel_thresh(image, kernel_size=9, lower_threshold=40, upper_threshold=220):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    hsl = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2HLS)
    sobel_l = cv2.Sobel(hsl[:, :, 1], cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_s = cv2.Sobel(hsl[:, :, 2], cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_mag_l = np.uint8(sobel_l / np.max(sobel_l) * 255)
    sobel_mag_s = np.uint8(sobel_s / np.max(sobel_s) * 255)
    binary_l, binary_s = np.zeros_like(sobel_l), np.zeros_like(sobel_s)
    binary_l[(sobel_mag_l >= lower_threshold) & (sobel_mag_l <= upper_threshold)] = 1
    binary_s[(sobel_mag_s >= lower_threshold) & (sobel_mag_s <= upper_threshold)] = 1
    binary_output = np.zeros_like(sobel_l)
    binary_output[(binary_l == 1) | (binary_s == 1)] = 1
    return binary_output


def yellow_thresh(image, lower_threshold=np.array([0, 50, 50]), upper_threshold=np.array([50, 255, 255])):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    binary_output = np.zeros_like(hls[:, :, 2])
    binary_output[(hls[:, :, 0] >= lower_threshold[0]) & (hls[:, :, 0] <= upper_threshold[0]) &
                  (hls[:, :, 1] >= lower_threshold[1]) & (hls[:, :, 1] <= upper_threshold[1]) &
                  (hls[:, :, 2] >= lower_threshold[2]) & (hls[:, :, 2] <= upper_threshold[2])] = 1
    binary_output = region_of_interest_color(binary_output)
    return binary_output


def white_thresh(image, lower_threshold=np.array([10, 0, 120]), upper_threshold=np.array([255, 150, 255])):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    binary_output = np.zeros_like(hls[:, :, 2])
    binary_output[(hls[:, :, 0] >= lower_threshold[0]) & (hls[:, :, 0] <= upper_threshold[0]) &
                  (hls[:, :, 1] >= lower_threshold[1]) & (hls[:, :, 1] <= upper_threshold[1]) &
                  (hls[:, :, 2] >= lower_threshold[2]) & (hls[:, :, 2] <= upper_threshold[2])] = 1
    binary_output = region_of_interest_color(binary_output)
    return binary_output


def binarize(image):
    binary_image_edge = canny_edge(image)
    binary_image_yellow = yellow_thresh(image)
    binary_image_white = white_thresh(image)
    binary_output = np.zeros_like(binary_image_edge)
    binary_output[(binary_image_edge == 1) | (binary_image_yellow == 1) | (binary_image_white == 1)] = 1
    return binary_output


if __name__ == '__main__':
    start_time = time()
    image = cv2.imread('test_image/real_time_driving_1_1_1.jpg')
    image_roi_edge = region_of_interest_edge(image)
    image_roi_color = region_of_interest_color(image)
    binary_image_edge = canny_edge(image)
    binary_image_yellow = yellow_thresh(image)
    binary_image_white = white_thresh(image)
    binary_image = binarize(image)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.figure()
    # plt.imshow(cv2.cvtColor(image_roi_edge, cv2.COLOR_BGR2RGB))
    # cv2.imwrite('output_image/image_roi_edge_1_1_1.png', image_roi_edge)
    # plt.figure()
    # plt.imshow(cv2.cvtColor(image_roi_color, cv2.COLOR_BGR2RGB))
    # cv2.imwrite('output_image/image_roi_color_1_1_1.png', image_roi_color)
    # plt.figure()
    plt.imshow(binary_image_edge, cmap='gray')
    # cv2.imwrite('output_image/binary_image_edge_1_1_1.png', binary_image_edge * 255)
    plt.figure()
    plt.imshow(binary_image_yellow, cmap='gray')
    # cv2.imwrite('output_image/binary_image_yellow_1_1_1.png', binary_image_yellow * 255)
    plt.figure()
    plt.imshow(binary_image_white, cmap='gray')
    # cv2.imwrite('output_image/binary_image_white_1_1_1.png', binary_image_white * 255)
    plt.figure()
    plt.imshow(binary_image, cmap='gray')
    # cv2.imwrite('output_image/binary_image_1_1_1.png', binary_image * 255)
    plt.figure()
    process_time = time() - start_time
    print(process_time)
    plt.show()
