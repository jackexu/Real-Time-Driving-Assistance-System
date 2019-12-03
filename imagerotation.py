import numpy as np
import cv2
import matplotlib.pyplot as plt

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    return result

if __name__ == '__main__':
    steering_wheel = cv2.imread('source/steering_wheel.png')
    plt.imshow(rotate_image(steering_wheel, 0))
    plt.show()