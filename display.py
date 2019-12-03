import numpy as np
import cv2
from birdsview import perspective_inv


def drawline_green(image, binary_image, lt_fit, rt_fit):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    y_range = np.linspace(0, binary_image.shape[0], num=50)
    # Fit new polynomials to x,y in world space
    lt_fitx = lt_fit[0] * y_range ** 2 + lt_fit[1] * y_range + lt_fit[2]
    rt_fitx = rt_fit[0] * y_range ** 2 + rt_fit[1] * y_range + rt_fit[2]
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_lt = np.array([np.transpose(np.vstack([lt_fitx, y_range]))])
    pts_rt = np.array([np.flipud(np.transpose(np.vstack([rt_fitx, y_range])))])
    pts = np.hstack((pts_lt, pts_rt))
    # Draw the lane onto the perspective blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (20, 240, 5))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    added_image = perspective_inv(color_warp)[0]
    result = cv2.addWeighted(image, 1, added_image, 0.2, 0)
    return result, color_warp


def drawline_red(image, binary_image, lt_fit, rt_fit):
    warp_zero = np.zeros_like(binary_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    y_range = np.linspace(0, binary_image.shape[0], num=50)
    lt_fitx = lt_fit[0] * y_range ** 2 + lt_fit[1] * y_range + lt_fit[2]
    rt_fitx = rt_fit[0] * y_range ** 2 + rt_fit[1] * y_range + rt_fit[2]
    pts_lt = np.array([np.transpose(np.vstack([lt_fitx, y_range]))])
    pts_rt = np.array([np.flipud(np.transpose(np.vstack([rt_fitx, y_range])))])
    pts = np.hstack((pts_lt, pts_rt))
    cv2.fillPoly(color_warp, np.int_([pts]), (100, 50, 220))
    added_image = perspective_inv(color_warp)[0]
    result = cv2.addWeighted(image, 1, added_image, 0.2, 0)
    return result, color_warp
