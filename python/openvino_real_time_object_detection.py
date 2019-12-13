# USAGE
# python openvino_real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2


important_objs = ['person']

class Box:
    def __init__(self, x1,y1,x2,y2 ):
        self.bottom = y2
        self.top = y1
        self.left = x1
        self.right = x2
class Point:
    def __init__(self, x,y):
        self.x = x
        self.y = y


def sitting_up_detection(detections):
    #{'person': Box, 'bed':Box}
    if len(detections) > 2: return False
    if 'person' not in detections or 'bed' not in detections: return False
    exit_left, exit_right, exit_top, exit_bottom = False, False, False, False
    person_pos = detections['person']
    #bed_pos = detections['manual_bed']
    bed_pos = detections['bed']
    delta = 0 # threshold of person bed overlap before detection
    if person_pos.x - delta < bed_pos.left:
        exit_left = True
    elif person_pos.x + delta > bed_pos.right:
        exit_right = True
    elif person_pos.y - delta > bed_pos.bottom:
        exit_bottom = True
    elif person_pos.y + delta < bed_pos.top:
        exit_top = True
    return exit_top or exit_bottom or exit_left or exit_right

def img_select_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        bed_point_list.append(Point(x,y))
        cv2.circle(param,(x,y),10,(255,0,0),-1)
        print(x,y)

def draw_bed(frame):
    global bed_point_list
    bed_point_list = []
    while True:
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', img_select_event, param=frame)
        cv2.imshow('image', frame)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
    return bed_point_list


def avg_box_pos(detected_obj_history):
    avg_box = Box(0,0,0,0)
    for b in detected_obj_history:
        avg_box.left += b.left/len(detected_obj_history)
        avg_box.right += b.right/len(detected_obj_history)
        avg_box.bottom += b.bottom/len(detected_obj_history)
        avg_box.top += b.top/len(detected_obj_history)

    avg_box.left = int(avg_box.left)
    avg_box.right = int(avg_box.right)
    avg_box.bottom = int(avg_box.bottom)
    avg_box.top = int(avg_box.top)
    return avg_box


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak detections")
ap.add_argument("-u", "--movidius", type=bool, default=0,
    help="boolean indicating if the Movidius should be used")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# specify the target device as the Myriad processor on the NCS
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
fps = FPS().start()
initialize_bed = True
bed_point_list = [[0,0],[0,0]]
detected_obj_history = []
frame_avg_num = 10
b = Box(0, 0, 0, 0)
detections = {}

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    bed_box = Box(0,0,0,0)
    if initialize_bed:
        bed_point_list = draw_bed(frame)
        initialize_bed = False
    bed_box = Box(bed_point_list[0].x, bed_point_list[0].y, bed_point_list[1].x, bed_point_list[1].y)
    detections['bed'] = bed_box
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        label = ''
        if confidence > args["confidence"]:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

            if idx == CLASSES.index('person'):
                obj_coords = box.astype('int')
                new_b = Box(obj_coords[0] - int(obj_coords[2]/2), obj_coords[1] - int(obj_coords[3]/2), obj_coords[0]+int(obj_coords[2]/2), obj_coords[1]+int(obj_coords[3]/2))
                detected_obj_history.append(new_b)
                if len(detected_obj_history) > frame_avg_num:
                    detected_obj_history = detected_obj_history[1:]
                b = avg_box_pos(detected_obj_history)
                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx],
                                             confidence * 100)

        person_point = Point(int((b.right + b.left) / 2), int((b.top + b.bottom) / 2))
        detections['person'] = person_point
        #  cv2.rectangle(frame, (b.left, b.top, b.right, b.bottom), (0, 255, 0), 2)
        cv2.circle(frame, (person_point.x, person_point.y), 10, (255, 0, 0), -1)
        is_sitting_up = sitting_up_detection(detections)
        (startX, startY, endX, endY) = b.astype("int")

        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (255,0,0), 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(frame, label, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()