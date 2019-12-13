import time
from ctypes import *
import math
import random
import cv2

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

lib = CDLL("/home/brandon/Codes/darknet/libdarknet.so", RTLD_GLOBAL)
#lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res
    
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

def capture_video(net, meta):
    cap = cv2.VideoCapture(0)
    temp_path = '/home/brandon/Codes/darknet/python/temp.jpg'
    temp_pathb = b'/home/brandon/Codes/darknet/python/temp.jpg'
    initialize_bed = True
    bed_point_list = []
    detected_obj_history = []
    frame_avg_num = 10
    b = Box(0, 0, 0, 0)
    detections = {}
    while(True):
        ret, frame = cap.read()
        if initialize_bed:
            bed_point_list = draw_bed(frame)
            initialize_bed = False
        cv2.rectangle(frame, bed_point_list[0], bed_point_list[1], (255,0,0), 5)
        bed_box = Box(bed_point_list[0][0], bed_point_list[0][1], bed_point_list[1][0], bed_point_list[1][1])
        detections['bed']= bed_box
        cv2.imwrite(temp_path, frame)
        frame_data = detect(net, meta, temp_pathb)
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
        cv2.circle(frame,(person_point.x,person_point.y),10,(255,0,0),-1)
        is_sitting_up = sitting_up_detection(detections)
        print(is_sitting_up)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        cv2.imshow('frame', frame)
        cv2.imwrite("image.jpg", gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    net = load_net(b"cfg/yolov3.cfg", b"yolov3.weights", 0)
    meta = load_meta(b"cfg/coco.data")
    capture_video(net, meta)
