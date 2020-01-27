#from send_message import send_message
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
    def __init__(self, p1,p2):
        self.bottom = p1.y
        self.top = p1.y
        self.left = p1.x
        self.right = p2.x


class Polygon:
    def __init__(self, top_left,bottom_left,top_right,bottom_right):
        self.top_left = top_left
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.bottom_right = bottom_right
        self.left_slope, self.left_intercept = self._calc_slope(bottom_left, top_left)
        self.right_slope, self.right_intercept = self._calc_slope(bottom_right, top_right)
    def _calc_slope(self,p2, p1):
        try:
            slope =(p2.y - p1.y) / float(p2.x - p1.x)
            intercept = p1.y-(slope*p1.x)
            return slope, intercept
        except:
            return -1,-1

    def get_right_x(self,p):
        right_x = (float(p.y- self.right_intercept)/self.right_slope)
        return right_x
    def get_left_x(self,p):
        left_x = (float(p.y-self.left_intercept)/self.left_slope)
        return left_x


class Point:
    def __init__(self, x,y):
        self.x = x
        self.y = y

class Obj:
    def __init__(self):
        self.detected_obj_history = []
        self.box = Box(Point(0,0),Point(0,0))
        self.polygon = Polygon(Point(0,0),Point(0,0), Point(0,0), Point(0,0))

class Person(Obj):
    def __init__(self):
        super(Person, self).__init__()

    def __str__(self):
        return 'Person'

class Bed(Obj):
    def __init__(self):
        super(Bed, self).__init__()

    def __str__(self):
        return 'Bed'

def sitting_up_detection(detections):
    #[Person, Person, Bed
    #{'person': Box, 'bed':Box}
    people = []
    bed = None
    person_found = False
    bed_found = False
    for d in detections:
        if 'Person' in str(d): people.append(d); person_found= True
        if 'Bed' in str(d): bed=d; bed_found = True
    if not person_found or not bed_found: return False
    exited = False
    for person in people:
        exit_left, exit_right, exit_top, exit_bottom = False, False, False, False
        person_pos = person.point
        #bed_pos = detections['manual_bed']
        bed_pos = bed.box
        delta = 0 # threshold of person bed overlap before detection
        if person_pos.x - delta < bed_pos.left:
            exit_left = True
        elif person_pos.x + delta > bed_pos.right:
            exit_right = True
        elif person_pos.y - delta > bed_pos.bottom:
            exit_bottom = True
        elif person_pos.y + delta < bed_pos.top:
            exit_top = True
        if exit_top or exit_bottom or exit_left or exit_right:
            exited = True
    return exited
def left_right_exit_detection(detections):
    #[Person, Person, Bed
    #{'person': Box, 'bed':Box}
    people = []
    bed = None
    person_found = False
    bed_found = False
    for d in detections:
        if 'Person' in str(d): people.append(d); person_found= True
        if 'Bed' in str(d): bed=d; bed_found = True
    if not person_found or not bed_found: return False
    exited = False
    for person in people:
        exit_left, exit_right = False, False 
        person_pos = person.point
        bed_pos = bed.polygon
        
        polygon_right_x, polygon_left_x = bed_pos.get_right_x(person_pos), bed_pos.get_left_x(person_pos)
        delta = 0 # threshold of person bed overlap before detection
        if person_pos.x - delta < polygon_left_x:
            exit_left = True
        elif person_pos.x + delta > polygon_right_x:
            exit_right = True
        if exit_left or exit_right:
            exited = True
    return exited


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
    avg_box = Box(Point(0,0), Point(0,0))
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
    #weightPath = "./yolov3_50000.weights"
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
    cap.set(3, 1280)
    cap.set(4, 720)
    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    bed_point_list = []
    detected_obj_history = []
    frame_avg_num = 10
    b = Box(Point(0,0), Point(0,0))
    situp_timer = time.time()  # starts when situp detected and reset after 10 mins
    first_time_detected = True
    detection_count = 0
    person_missing_count = 0
    def get_frame():
        ret, frame_read = cap.read()
        #frame_read = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        w = int(frame_read.shape[0]/2)
        h = int(frame_read.shape[1]/2)
        size = int(416/2)
        frame_read = frame_read[w-size:w+size,h-size:h+size]
        frame_resized = cv2.resize(frame_read,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)
        return frame_resized
    frame = get_frame()
    bed_point_list = draw_bed(frame)
    bed = Bed()
    try:
        # depends on whether user generates polygon or box for exit
        bed.box = Box(bed_point_list[0], bed_point_list[1])
        bed.polygon = Polygon(bed_point_list[0],bed_point_list[1],bed_point_list[2],bed_point_list[3])
    except Exception as e:
        print(e)
    person_missing_time = time.time()
    while True:
        prev_time = time.time()
        frame = get_frame()
        #frame[:,:,0]=frame[:,:,2]
        #frame[:,:,1]=frame[:,:,2]
        darknet.copy_image_from_bytes(darknet_image,frame.tobytes())
        frame_data = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        person_found = False
        people_list= []
        for content in frame_data:
            key = content[0].decode('utf-8')
            if key == 'person':
                person_found = True
                person_missing_time = time.time()
                obj_coords = [int(c) for c in content[2]]
                new_b = Box(Point(obj_coords[0] - int(obj_coords[2]/2), obj_coords[1] - int(obj_coords[3]/2)), Point(obj_coords[0]+int(obj_coords[2]/2), obj_coords[1]+int(obj_coords[3]/2)))
                p = Person()
                p.detected_obj_history.append(new_b)
                if len(p.detected_obj_history) > frame_avg_num:
                    p.detected_obj_history = p.detected_obj_history[1:]
                p.box = avg_box_pos(p.detected_obj_history)
                p.point = Point(int((p.box.right + p.box.left) / 2), int((p.box.top + p.box.bottom) / 2))
                people_list.append(p)

        detections = []
        detections.extend(people_list)
        detections.append(bed)
        is_sitting_up = left_right_exit_detection(detections)
        situp_time_elapsed = time.time() - situp_timer
        is_sitting_up_detection = is_sitting_up and (situp_time_elapsed >= 600 or first_time_detected)
        is_missing_detection = (time.time() - person_missing_time > 5)   
        #print('missing ', is_missing_detection)
        #print('sitting ', is_sitting_up)
        print(is_missing_detection or is_sitting_up)
        color = (0,255,0)
        if is_missing_detection or is_sitting_up:
            color = (0,0,255)
            detection_count += 1
            cv2.imwrite('python/detection_images/{}.png'.format(detection_count), frame)
            #send_message("+15712513711")
            situp_timer = time.time()
            first_time_detected = False
        for person in people_list:
            cv2.circle(frame,(person.point.x,person.point.y),10,color,-1)
        cv2.circle(frame, (bed.polygon.top_left.x, bed.polygon.top_left.y), 10, color, -1)
        cv2.circle(frame, (bed.polygon.top_right.x, bed.polygon.top_right.y), 10, color, -1)
        cv2.circle(frame, (bed.polygon.bottom_left.x, bed.polygon.bottom_left.y), 10, color, -1)
        cv2.circle(frame, (bed.polygon.bottom_right.x, bed.polygon.bottom_right.y), 10, color, -1)
        #cv2.rectangle(frame, (bed.box.left, bed.box.top), (bed.box.right, bed.box.bottom), (0, 255, 0), 2)
        #print(1/(time.time()-prev_time))
        cv2.imshow('Demo', frame)
        cv2.waitKey(3)
    cap.release()

if __name__ == "__main__":
    YOLO()
