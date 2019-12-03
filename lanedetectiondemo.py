import cv2
from lanedetection import lane_detection


def video_demo(video_filepath):
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out_video = cv2.VideoWriter('output_lane.avi', fourcc, 20.0, (480, 270))
    count = 0
    average_steer_list = []
    no_detection_list = []
    # Read and copy video
    video = cv2.VideoCapture(video_filepath)
    while video.isOpened():
        # Read in frame
        ret, image = video.read()
        # If frame read is successful
        if ret:
            out, _, _ = lane_detection(image, count, average_steer_list, no_detection_list)
            out = cv2.resize(out, (960, 540))
            #out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        # If frame read is unsuccessful
        else:
            break
        count += 1
        print(count)
        out_video.write(out)
        # Show combined image
        cv2.imshow("lane_detection", out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    out_video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_demo('test_video/real_time_driving_2_1.mp4')
