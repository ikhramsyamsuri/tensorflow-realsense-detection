import warnings
import numpy as np
from object_detection.utils import label_map_util as CategoryLabel
import time
from base.mask_rcnn import MaskRCNN
from base.realsense_camera import RealsenseCamera
import cv2
import argparse
import tensorflow as tf
import os
# Suppress TensorFlow logging (1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Load the model
# ===============================================================================================================

# Suppress Matplotlib warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow logging (2)
tf.get_logger().setLevel('ERROR')

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Folder that the Saved Model is Located In',
                    default='models')
parser.add_argument('--labels', help='Where the Labelmap is Located',
                    default='models/label_map.pbtxt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)

args = parser.parse_args()
# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = args.model

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = args.labels

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = float(args.threshold)


PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))


# Load label map data (for plotting)
# ==================================================================================================================================

category_index = CategoryLabel.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                   use_display_name=True)
# Load Realsense camera
rs = RealsenseCamera()
mrcnn = MaskRCNN()

print('Running inference for Webcam', end='')

# Initialize Webcam
videostream = cv2.VideoCapture(0)
ret = videostream.set(3, 1280)
ret = videostream.set(4, 720)

# Deep camera stream
deepBol, bgr_frame, depth_frame = rs.get_frame_stream()

# Detect Object
boxes, classes, contours, centers = mrcnn.detect_objects_mask(
    bgr_frame)

while True:

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = videostream.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_expanded = np.expand_dims(frame_rgb, axis=0)
    imH, imW, _ = frame.shape

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(frame)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(
        np.int64)

    # SET MIN SCORE THRESH TO MINIMUM THRESHOLD FOR DETECTIONS

    detections['detection_classes'] = detections['detection_classes'].astype(
        np.int64)
    scores = detections['detection_scores']
    boxes = detections['detection_boxes']
    classes = detections['detection_classes']
    count = 0

    for i in range(len(scores)):
        if ((scores[i] > MIN_CONF_THRESH) and (scores[i] <= 1.0)):
            # increase count
            count += 1
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            # Draw label
            # Look up object name from "labels" array using class index
            object_name = category_index[int(classes[i])]['name']

            # Jarak deph camera using rectacgle area
            depth_mm = depth_frame[ymin, xmin]

            # Label Barang, Score, Jarak
            label = '%s: %d%%,  %d CM' % (object_name, int(
                scores[i]*100), depth_mm / 10)  # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size

            # Make sure not to draw label too close to top of window
            label_ymin = max(ymin, labelSize[1] + 10)
            # Draw white box to put label text in
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (
                xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)

            # Draw label text
            cv2.putText(frame, label, (xmin, label_ymin-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.putText(frame, 'Objects Detected : ' + str(count), (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (70, 235, 52), 2, cv2.LINE_AA)

    cv2.imshow('Objects Detector', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
print("Done")
