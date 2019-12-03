import numpy as np
import tensorflow as tf


from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
# from grabscreen import grab_screen
import cv2

# ## Object detection imports
# Here are the imports from the object detection module.

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

video = cv2.VideoCapture('test_video/real_time_driving_1_1.mp4')

def distance_function(cap):
# # Model preparation
# Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_FROZEN_GRAPH = 'object_detection/utils/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = 'object_detection/utils/data/mscoco_label_map.pbtxt'

    NUM_CLASSES = 90
# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)


# ## Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        with tf.Session(graph=detection_graph) as sess:
            while True:
      #screen = cv2.resize(grab_screen(region=(0,40,1280,745)), (WIDTH,HEIGHT))
      # screen = cv2.resize(grab_screen(region=(0,40,1280,745)), (800,450))
                image_np = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
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
                    feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

                for i,b in enumerate(boxes[0]):
        #                 car                    bus                  truck
                    if classes[0][i] == 3 or classes[0][i] == 6 or classes[0][i] == 8:
                        if scores[0][i] >= 0.5:
                            mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                            mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                            apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                            cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                            if apx_distance <=0.5:
                                if mid_x > 0.3 and mid_x < 0.7:
                                    cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)


                # cv2.imshow('window',cv2.resize(image_np,(800,450)))
                # if cv2.waitKey(25) & 0xFF == ord('q'):
                #     cv2.destroyAllWindows()
                #     break
if __name__ == '__main__':
    distance_function(video)