import cv2


def img_select_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        bed_point_list.extend([x,y])
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

def main():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    bed_point_list = draw_bed(frame)
    with open('python/bed_points.txt', 'w+') as f:
        f.write(str(bed_point_list))


if __name__ == '__main__':
    main()
