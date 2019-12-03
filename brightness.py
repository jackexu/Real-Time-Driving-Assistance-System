import cv2
import glob
import math
from PIL import Image, ImageStat
from time import time


def perceived_brightness(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    stat = ImageStat.Stat(img)
    r, g, b = stat.mean
    return math.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))


if __name__ == '__main__':
    test_images = glob.glob('test_image/*.jpg')
    for test_image in test_images:
        start_time = time()
        image = cv2.imread(test_image)
        brightness = perceived_brightness(image)
        process_time = time() - start_time
        print(brightness)
        print(process_time)
        print('')
