import cv2

def distance_estimation(image_np, scores, classes, boxes):
    for i, b in enumerate(boxes[0]):
        #  1 = person; 2 = bicycle; 3 = car; 4 = motorcycle;
        if classes[0][i] == 1 or classes[0][i] == 2 or classes[0][i] == 3 or classes[0][i] == 4:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1] + boxes[0][i][3]) / 2
                mid_y = (boxes[0][i][0] + boxes[0][i][2]) / 2
                # print(boxes[0][i]) [0.6581085  0.53408283 0.6880827  0.55805546]
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1])) ** 4), 2)
                distance_num_size = 1
                # cv2.putText(image_np,'{}'.format(apx_distance),(int(mid_x * 1920),int((mid_y-0.025)*1080)),
                #             cv2.FONT_HERSHEY_SIMPLEX, distance_num_size, (255, 255, 255), 1)
                cv2.putText(image_np, '{}'.format(apx_distance),
                            (int(mid_x * 1920), int((boxes[0][i][0]) * 1080 - 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, distance_num_size, (0, 0, 255), 2)
                if apx_distance <= 0.9:
                    if mid_x > 0.49 and mid_x < 0.51:
                        cv2.putText(image_np, 'Warning!!!', (int((mid_x - 0.02) * 1920), int(mid_y * 1080)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        # 6 = bus; 8 = truck; Since they are bigger on screen, adjust distance critical value
        elif classes[0][i] == 6 or classes[0][i] == 8:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1] + boxes[0][i][3]) / 2
                mid_y = (boxes[0][i][0] + boxes[0][i][2]) / 2
                # Change to use y difference for large vehicle.
                apx_distance = round(((1 - (boxes[0][i][2] - boxes[0][i][0])) ** 3), 2)
                distance_num_size = 1.0
                cv2.putText(image_np, '{}'.format(apx_distance),
                            (int(mid_x * 1920), int((boxes[0][i][0]) * 1080 - 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, distance_num_size, (0, 0, 255), 2)
                if apx_distance <= 0.5:
                    # (0.4,0.7) would cause _2_1 warning.
                    if mid_x > 0.47 and mid_x < 0.53:
                        cv2.putText(image_np, 'Warning!!!', (int((mid_x - 0.02) * 1920), int(mid_y * 1080)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    return image_np