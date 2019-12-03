import cv2
import numpy as np
from imagerotation import rotate_image
import matplotlib.pyplot as plt
from brightness import perceived_brightness
from binarization import binarize
from birdsview import perspective, perspective_inv
from linefit import poly_second, start_fit, continue_fit, check_fit
from display import drawline_green, drawline_red
from calculation import curvature_radius, center_offset, steering_angle
from time import time


def lane_detection(image, count, average_steer_list, no_detection_list):
    global left_fit
    global right_fit
    global left_fit_prev
    global right_fit_prev
    binary_image = binarize(image)
    perspective_image = perspective(binary_image)[0]
    if count == 0:
        left_fit, right_fit, out, hist = start_fit(perspective_image)
    else:
        try:
            left_fit, right_fit = continue_fit(perspective_image, left_fit, right_fit)
            no_detection_list.append(0)
        except:
            print('No Detection')
            no_detection_list.append(1)
            left_fit, right_fit = left_fit_prev, right_fit_prev
    check_status = check_fit(perspective_image, left_fit, right_fit)
    if check_status == True:
        left_fit_prev, right_fit_prev = left_fit, right_fit
        no_detection_list.append(0)
    else:
        if count == 0:
            pass
        else:
            left_fit, right_fit = left_fit_prev, right_fit_prev
            print('No Detection')
            no_detection_list.append(1)

    brightness = perceived_brightness(image)
    average_rad = curvature_radius(perspective_image, left_fit, right_fit)
    steer = steering_angle(average_rad)
    average_steer_list.append(steer)
    if len(average_steer_list) >= 3 and np.abs(average_steer_list[-1] - average_steer_list[-2]) > 1:
        steer = np.average(average_steer_list[-3:], weights=range(1, 4, 1))
    center = center_offset(left_fit, right_fit)
    text_fra = "Frame: " + str(count + 1)
    text_brt = "Perceived Brightness: " + str(round(brightness, 2))
    text_ang = "Steering Angle: " + str(round(steer, 2))
    if len(no_detection_list) >= 5 and np.sum(no_detection_list[-5])/5 >= 0.2:
        text_nod = "NO DETECTION!"
    else:
        text_nod = "LANES DETECTED"
    if np.abs(average_rad) >= 5000:
        text_cur = "Curvature Radius: --m"
    else:
        text_cur = "Curvature Radius: " + str(round(average_rad, 2)) + "m"
    if center > 0.1:
        if center > 0.5:
            text_cen = "Center Offset: " + str(center) + "m"
            text_str = "Lane Keeping Status: " + "CAUTION! LEFT!"
            img_merge, img_birds = drawline_red(image, perspective_image, left_fit, right_fit)
        else:
            text_cen = "Center Offset: " + str(center) + "m"
            text_str = "Lane Keeping Status: " + "SLIGHTLY LEFT"
            img_merge, img_birds = drawline_red(image, perspective_image, left_fit, right_fit)
    elif center < -0.1:
        if center < -0.5:
            text_cen = "Center Offset: " + str(center) + "m"
            text_str = "Lane Keeping Status: " + "CAUTION! RIGHT!"
            img_merge, img_birds = drawline_red(image, perspective_image, left_fit, right_fit)
        else:
            text_cen = "Center Offset: " + str(center) + "m"
            text_str = "Lane Keeping Status: " + "SLIGHTLY RIGHT"
            img_merge, img_birds = drawline_red(image, perspective_image, left_fit, right_fit)
    else:
        text_cen = "Center Offset: " + str(center) + "m"
        text_str = "Lane Keeping Status: " + "GOOD"
        img_merge, img_birds = drawline_green(image, perspective_image, left_fit, right_fit)

    if "!" in text_cen:
        cv2.putText(img_merge, text_cen, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(img_merge, text_cen, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 100, 50), 2)

    textsize = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    textX = int((img_merge.shape[1] - textsize[0]) / 2)
    cv2.putText(img_merge, text_str, (textX, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    textsize = cv2.getTextSize(text_nod, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    textX = int((img_merge.shape[1] - textsize[0]) / 2)
    if text_nod == "NO DETECTION!":
        cv2.putText(img_merge, text_nod, (textX, 850), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    else:
        cv2.putText(img_merge, text_nod, (textX, 850), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.putText(img_merge, text_fra, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 100, 50), 2)
    cv2.putText(img_merge, text_brt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 100, 50), 2)
    cv2.putText(img_merge, text_cur, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 100, 50), 2)
    cv2.putText(img_merge, text_ang, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 100, 50), 2)

    steering_wheel = cv2.imread('source/steering_wheel.png')
    steering_wheel = rotate_image(steering_wheel, -steer)
    steering_wheel = cv2.resize(steering_wheel, (300, 300))
    h, w = steering_wheel.shape[:2]
    img_merge[10:10 + h, 1600:1600 + w] = steering_wheel
    return img_merge, left_fit, right_fit


if __name__ == '__main__':
    start_time = time()
    image = cv2.imread('test_image/real_time_driving_1_1_1.jpg')
    lane_detection, lt_fit, rt_fit = lane_detection(image, 0, [], [])
    perspect = perspective(lane_detection)[0]
    plt.imshow(lane_detection)
    plt.figure()

    y_range = np.linspace(0, perspect.shape[0], num = 100)
    lt_fit_x = lt_fit[0] * y_range ** 2 + lt_fit[1] * y_range + lt_fit[2]
    rt_fit_x = rt_fit[0] * y_range ** 2 + rt_fit[1] * y_range + rt_fit[2]
    plt.plot(lt_fit_x, y_range, color='red')
    plt.plot(rt_fit_x, y_range, color='red')
    plt.imshow(perspect)
    plt.figure()
    #cv2.imwrite('output_image/123.png', lane_detection)
    process_time = time() - start_time
    print(process_time)
    plt.show()

