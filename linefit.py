import numpy as np
import cv2
import matplotlib.pyplot as plt
from binarization import binarize
from birdsview import perspective
from time import time


def poly_second(lt_x, lt_y, rt_x, rt_y):
    ##Fit a second order polynomial to lt and rt lane from birdsview image
    ##If there are no points on the lt
    if len(lt_x) == 0:
        lt_ = []
    ##If there are points found on the lt from moving windows
    else:
        ##Weights to apply to the y-coordinates of the sample points
        start, stop = 1.0, 0.6
        step = -(start - stop) / len(lt_y)
        try:
            lt_ = np.polyfit(lt_y, lt_x, 2, full = True, w = np.arange(start, stop, step))
        except:
            lt_ = np.polyfit(lt_y, lt_x, 2, full = True)
    ##If there are no points on the rt
    if len(rt_x) == 0:
        rt_ = []
    ##If there are points found on the rt from moving windows
    else:
        ##Weights to apply to the y-coordinates of the sample points
        start, stop = 1.0, 0.6
        step = -(start - stop) / len(rt_y)
        try:
            rt_ = np.polyfit(rt_y, rt_x, 2, full = True, w = np.arange(start, stop, step))
        except:
            rt_ = np.polyfit(rt_y, rt_x, 2, full=True)
    ##If residuals of lt and rt are very different
    if len(lt_x) != 0 and len(rt_x) != 0 and lt_[0][0] * rt_[0][0] < 0:
        ##If rt lane fit is more accurate than lt
        if lt_[1][0] / rt_[1][0] > 3:
            ##Adjust weights
            start, stop = 1.0, 0.1
            step = -(start - stop) / len(lt_y)
            try:
                lt_ = np.polyfit(lt_y, lt_x, 2, full = True, w = np.arange(start, stop, step))
            except:
                pass
        ##If lt lane fit is more accurate than rt
        elif rt_[1][0] / lt_[1][0] > 3:
            ##Adjust weights
            start, stop = 1.0, 0.1
            step = -(start - stop) / len(rt_y)
            try:
                rt_ = np.polyfit(rt_y, rt_x, 2, full = True, w = np.arange(start, stop, step))
            except:
                pass
        else:
            pass
    ##Output coefficients of lt and rt lane fits
    lt_fit = lt_[0]
    rt_fit = rt_[0]
    return lt_fit, rt_fit

def start_fit(perspective, histstart = 3/10, totalwindows = 10, margin = 150, minthresh = 80):
    out = np.dstack((perspective, perspective, perspective)) * 255
    image_height = perspective.shape[0]
    lt_index = []
    rt_index = []
    # Identify the x and y positions of all nonzero pixels in the image
    x_nonzero = np.array(perspective.nonzero()[1])
    y_nonzero = np.array(perspective.nonzero()[0])
    #Histogram of the bottom histstart of the image
    hist = np.sum(perspective[int(image_height * (1 - histstart)):, :], axis = 0)
    #lt and rt peaks of the hist
    lt_start = np.argmax(hist[:int(hist.shape[0] / 2)])
    rt_start = int(hist.shape[0] / 2) + np.argmax(hist[int(hist.shape[0] / 2):])
    # Current positions to be updated for each window
    lt_current = lt_start
    rt_current = rt_start
    #Window height
    window_height = int(image_height / totalwindows)
    for window in range(totalwindows):
        ##Window lines
        x_lt_low, x_lt_high = lt_current - int(margin / 2), lt_current + int(margin / 2)
        x_rt_low, x_rt_high = rt_current - int(margin / 2), rt_current + int(margin / 2)
        y_low  = image_height - window * window_height - window_height
        y_high = image_height - window * window_height
        # cv2.rectangle(out,(x_lt_low,y_low),(x_lt_high,y_high),(0,255,0), 2)
        # cv2.rectangle(out,(x_rt_low,y_low),(x_rt_high,y_high),(0,255,0), 2)
        #Identify the nonzero pixels in x and y within the window
        lt_index_temp = ((y_nonzero >= y_low) & (y_nonzero < y_high) & (x_nonzero >= x_lt_low) & (x_nonzero < x_lt_high)).nonzero()[0]
        rt_index_temp = ((y_nonzero >= y_low) & (y_nonzero < y_high) & (x_nonzero >= x_rt_low) & (x_nonzero < x_rt_high)).nonzero()[0]
        # Append these indices to the lists
        lt_index.append(lt_index_temp)
        rt_index.append(rt_index_temp)
        #Update the location of windows for the next step
        if len(lt_index_temp) > minthresh:
            lt_current = int(np.mean(x_nonzero[lt_index_temp]))
        else:
            pass
        if len(rt_index_temp) > minthresh:
            rt_current = int(np.mean(x_nonzero[rt_index_temp]))
        else:
            pass
    # Concatenate the arrays of indices
    lt_index, rt_index = np.concatenate(lt_index), np.concatenate(rt_index)
    # Extract lt and rt line pixel positions
    lt_x, lt_y = x_nonzero[lt_index], y_nonzero[lt_index]
    rt_x, rt_y = x_nonzero[rt_index], y_nonzero[rt_index]

    lt_fit, rt_fit = poly_second(lt_x, lt_y, rt_x, rt_y)
    return lt_fit, rt_fit, out, hist

def continue_fit(perspective, lt_fit, rt_fit, margin = 100):
    x_nonzero = np.array(perspective.nonzero()[1])
    y_nonzero = np.array(perspective.nonzero()[0])

    lt_index = ((x_nonzero > (lt_fit[0] * (y_nonzero ** 2) + lt_fit[1] * y_nonzero + lt_fit[2] - int(margin / 2))) &
                (x_nonzero < (lt_fit[0] * (y_nonzero ** 2) + lt_fit[1] * y_nonzero + lt_fit[2] + int(margin / 2))))
    rt_index = ((x_nonzero > (rt_fit[0] * (y_nonzero ** 2) + rt_fit[1] * y_nonzero + rt_fit[2] - int(margin / 2))) &
                (x_nonzero < (rt_fit[0] * (y_nonzero ** 2) + rt_fit[1] * y_nonzero + rt_fit[2] + int(margin / 2))))

    lt_x, lt_y = x_nonzero[lt_index], y_nonzero[lt_index]
    rt_x, rt_y = x_nonzero[rt_index], y_nonzero[rt_index]

    lt_fit_updated, rt_fit_updated = poly_second(lt_x, lt_y, rt_x, rt_y)

    return lt_fit_updated, rt_fit_updated

def check_fit(perspective, lt_fit, rt_fit, width_thresh_lower=600, width_thresh_upper=850, tangent_thresh=0.5):
    check_status = True
    lt_fit_x = lt_fit[0] * (perspective.shape[0]/2) ** 2 + lt_fit[1] * (perspective.shape[0]/2) + lt_fit[2]
    rt_fit_x = rt_fit[0] * (perspective.shape[0]/2) ** 2 + rt_fit[1] * (perspective.shape[0]/2) + rt_fit[2]
    width = np.abs(lt_fit_x - rt_fit_x)
    print(width_thresh_lower, width_thresh_upper, width)
    if width < width_thresh_lower or width > width_thresh_upper:
        check_status = False
    for i in np.arange(0.8, -0.2, -0.2):
        lt_fit_tangent = 2 * lt_fit[0] * perspective.shape[0] * i + lt_fit[1]
        rt_fit_tangent = 2 * rt_fit[0] * perspective.shape[0] * i + rt_fit[1]
        tandiff = np.abs(lt_fit_tangent - rt_fit_tangent)
        if tandiff > tangent_thresh:
            check_status = False
    return check_status


if __name__ == '__main__':
    start_time = time()
    image = cv2.imread('test_image/real_time_driving_3_1_1.jpg')
    binary_image = binarize(image)
    perspective_image = perspective(binary_image)[0]
    lt_fit, rt_fit, out, hist = start_fit(perspective_image)
    print(check_fit(perspective_image, lt_fit, rt_fit))
    y_range = np.linspace(0, perspective_image.shape[0], num = 100)
    lt_fit_x = lt_fit[0] * y_range ** 2 + lt_fit[1] * y_range + lt_fit[2]
    rt_fit_x = rt_fit[0] * y_range ** 2 + rt_fit[1] * y_range + rt_fit[2]

    #plt.plot(hist)
    #plt.set_cmap('hot')
    #plt.xlim(left=0, right=perspective_image.shape[1])
    #plt.ylim(bottom=0, top=200)
    #plt.savefig('output_image/hist_2_2_1.png', dpi = 1000)
    #plt.figure()

    #plt.imshow(out)
    #plt.axis('off')
    #plt.savefig('output_image/windows_noline_2_2_1.png', dpi = 1000)
    #plt.figure()

    plt.plot(lt_fit_x, y_range, color='red')
    plt.plot(rt_fit_x, y_range, color='red')
    plt.imshow(out)
    plt.axis('off')
    plt.savefig('output_image/windows_nowindow_3_1_1.png', dpi = 1000)
    plt.figure()

    process_time = time() - start_time
    print(process_time)
    plt.show()
