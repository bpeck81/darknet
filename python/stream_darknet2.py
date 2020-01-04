from send_message import send_message
from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img


netMain = None
metaMain = None
altNames = None


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
        bed_point_list.append((x,y))
        cv2.circle(param,(x,y),10,(255,0,0),-1)
        #print(x,y)

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


def YOLO():

    global metaMain, netMain, altNames
    configPath = "./cfg/yolov3.cfg"
    weightPath = "./yolov3.weights"
    metaPath = "./cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture("test.mp4")
    cap.set(3, 1280)
    cap.set(4, 720)
    out = cv2.VideoWriter(
        "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
        (darknet.network_width(netMain), darknet.network_height(netMain)))
    #print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    initialize_bed = True
    bed_point_list = []
    detected_obj_history = []
    frame_avg_num = 2
    b = Box(0, 0, 0, 0)
    detections = {}
    situp_timer = time.time()  # starts when situp detected and reset after 10 mins
    first_time_detected = True
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        if initialize_bed:
            bed_point_list = draw_bed(frame_resized)
            initialize_bed = False
        cv2.rectangle(frame_resized, bed_point_list[0], bed_point_list[1], (255,0,0), 5)
        bed_box = Box(bed_point_list[0][0], bed_point_list[0][1], bed_point_list[1][0], bed_point_list[1][1])
        detections['bed']= bed_box
        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        frame_data = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)

        for content in frame_data:
            key = content[0].decode('utf-8')
            if key == 'person':
                obj_coords = [int(c) for c in content[2]]
                new_b = Box(obj_coords[0] - int(obj_coords[2]/2), obj_coords[1] - int(obj_coords[3]/2), obj_coords[0]+int(obj_coords[2]/2), obj_coords[1]+int(obj_coords[3]/2))
                detected_obj_history.append(new_b)
                if len(detected_obj_history) > frame_avg_num:
                    detected_obj_history = detected_obj_history[1:]
                b = avg_box_pos(detected_obj_history)
        person_point = Point(int((b.right +b.left)/2), int((b.top+b.bottom)/2))
        detections['person'] = person_point
      #  cv2.rectangle(frame, (b.left, b.top, b.right, b.bottom), (0, 255, 0), 2)
        is_sitting_up = sitting_up_detection(detections)
        #print(is_sitting_up)
        situp_time_elapsed = time.time() - situp_timer
        if is_sitting_up and (situp_time_elapsed >= 600 or first_time_detected):
            send_message()
            situp_timer = time.time()
            first_time_detected = False
    #    image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        cv2.circle(image,(person_point.x,person_point.y),10,(255,0,0),-1)
        #print(1/(time.time()-prev_time))
        cv2.imshow('Demo', image)
        cv2.waitKey(3)
    cap.release()
    out.release()

if __name__ == "__main__":
    YOLO()
