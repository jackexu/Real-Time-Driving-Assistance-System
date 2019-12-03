import cv2
import numpy as np
import tensorflow as tf

from lanedetection import lane_detection
from object_detection.utils.label_map_util import create_category_index_from_labelmap
from object_detection.utils.visualization_utils import visualize_boxes_and_labels_on_image_array
from distanceestimation import distance_estimation

cap = cv2.VideoCapture('test_video/real_time_driving_2_2.mp4')

# Define the codec and create VideoWriter object: DIVX or XVID
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output_1_1.avi',fourcc, 30.0, (1920,1080))

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = 'object_detection/utils/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'object_detection/utils/data/mscoco_label_map.pbtxt'

# Using labels to create category index
category_index = create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name = True)

"""
# 1: {'id': 1, 'name': 'person'},
# 2: {'id': 2, 'name': 'bicycle'},
# 3: {'id': 3, 'name': 'car'},
# 4: {'id': 4, 'name': 'motorcycle'},
# 5: {'id': 5, 'name': 'airplane'},
# 6: {'id': 6, 'name': 'bus'},
# 7: {'id': 7, 'name': 'train'},
# 8: {'id': 8, 'name': 'truck'},
# 9: {'id': 9, 'name': 'boat'},
# 10: {'id': 10, 'name': 'traffic light'},
# 11: {'id': 11, 'name': 'fire hydrant'},
# 13: {'id': 13, 'name': 'stop sign'},
# 14: {'id': 14, 'name': 'parking meter'},
# 15: {'id': 15, 'name': 'bench'},
"""

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name = '')

    with tf.Session(graph=detection_graph) as sess:
        count = 0
        average_steer_list = []
        no_prediction_list = []
        while cap.isOpened():
            ret, image_np = cap.read()
            if ret:
                image_np_init = np.copy(image_np)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis = 0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict = {image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates = True,
                    line_thickness = 8)
                distance_estimation(image_np, scores, classes, boxes)
                # Add lane detection function
                image_ld, _, _ = lane_detection(image_np_init, count, average_steer_list, no_prediction_list)
                result = cv2.addWeighted(image_np, 0.55, image_ld, 0.55, 0)
            else:
                break
            count += 1

            # write the flipped frame
            out.write(result)
            
            cv2.imshow('Real-Time Driving Assistance System', result)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
