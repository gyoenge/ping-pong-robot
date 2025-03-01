import cv2
import numpy as np
import multiprocessing as mp
import socket
import struct 
import time
import sys
sys.path.append("..")
from ControlSystem.motor_controller import MC  

def capture_frame(queue):
    cap = cv2.VideoCapture()
    
    cap.open(0 + cv2.CAP_DSHOW)
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        queue.put(frame)
    cap.release()
def capture_frame_next(queue, queue_out) :
    # matrix1 = np.array([[1.0690399877867667, -0.01005366759674067, -124.39402917447335], [0.03689230874407605, 1.06770681776973, -76.92046373139854], [7.385240554183268e-05, 6.095526219810171e-05, 1.0]])
    # w1 = 502
    # h1 = 330
    matrix1 = np.array([[1.0807503551370208, -0.01699293011221724, -124.19452901815143], [0.0438744002135565, 1.079310245253488, -79.56183734726324], [0.00010577725072932761, 3.995998840073267e-05, 1.0]])
    w1 = 499
    h1 = 334
    while True :
        frame = queue.get()
        frame = cv2.warpPerspective(frame, matrix1, (w1, h1))
        queue_out.put(frame)

def extract_orange(queue_in, queue_out):
    level = 0
    lower_orange = np.array([0, 70, 70])
    upper_orange = np.array([30, 255, 255])
    while True:
        frame = queue_in.get()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        cv2.imshow('Mask', mask)
        countN = cv2.countNonZero(mask)
        if countN > 0 and level < 2:
            queue_out.put(mask)
            level += 1
        elif countN == 0 :
            level = 0
            queue_out.put(None)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

def print_center(queue, queue_out):
    
    while True:
        mask = queue.get()
        if mask is not None :
            M = cv2.moments(mask)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            queue_out.put((cX, cY))
            print(f"Center of mass is at ({cX}, {cY})")
        else :
            queue_out.put(None)

def estimate_line(queue, queue_out) :
    pre_pos = None
    while True :
        pos = queue.get()
        if pre_pos is not None and pos is not None :
            queue_out.put([pre_pos, pos])
            print(pre_pos, pos)
            pre_pos = pos
        else :
            pre_pos = pos

def get_linear_position(queue) :
    y_len = 262.3*(480/137)
    address = '192.168.0.106'
    port = 12345
    max_pos = 141
    init_pos = 71
    curr_pos = 71
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    message = 71
    mc = MC("COM7")
    mc.motion_1()
    time.sleep(1)
    network_message = struct.pack('!i', message)
    sent = sock.sendto(network_message, (address, port))

    while True :
        poss = queue.get()
        if (poss[1][1] > poss[0][1]) :
            LX = poss[0][0] + ((poss[1][0]-poss[0][0])/(poss[1][1]-poss[0][1]))*(y_len-poss[0][1])
            mm = (141-141*LX/640)
            message = int(mm - curr_pos)
            
            if mm < 80 :
                message = max(message-20, -71)
            elif 80 <= mm and mm <= 100 :
                message = 0
            elif 100 < mm and mm < 130 :
                message = min(message-10, 141)
            print("move to :", message + curr_pos)
            #if message < -50 :
            #    message = max(message*2, -71)
            #elif message < 0 :
            #    message = max(message*3, -71)
            #elif 0 < message and message < 20 :
            #    message = min(int(message*0.6), 71)
            if curr_pos + message > max_pos:
                message = max_pos - curr_pos
            elif curr_pos + message < 0:
                message = -curr_pos

            try:
                network_message = struct.pack('!i', message)
                sent = sock.sendto(network_message, (address, port))
                curr_pos += message
                mc.motion_3(0.1)
                time.sleep(2)
                network_message = struct.pack('!i', -message)
                sent = sock.sendto(network_message, (address, port))
                curr_pos -= message


            except:
                print("Linear-control-error")
        
if __name__ == "__main__":
    queue1 = mp.Queue(maxsize=5)
    queue1_1 = mp.Queue(maxsize=5)
    queue2 = mp.Queue(maxsize=5)
    queue3 = mp.Queue(maxsize=5)
    queue4 = mp.Queue(maxsize=5)

    process1 = mp.Process(target=capture_frame, args=(queue1_1,))
    process1_1 = mp.Process(target=capture_frame_next, args=(queue1_1, queue1, ))
    process2 = mp.Process(target=extract_orange, args=(queue1, queue2,))
    process3 = mp.Process(target=print_center, args=(queue2, queue3, ))
    process4 = mp.Process(target=estimate_line, args=(queue3, queue4, ))
    process5 = mp.Process(target=get_linear_position, args=(queue4,))

    try :
        process1.start()
        process1_1.start()
        process2.start()
        process3.start()
        process4.start()
        process5.start()

        process1.join()
        process2.join()
        process3.join()
        process4.join()
        process5.join()
    except :
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        address = '192.168.0.106'
        port = 12345
        network_message = struct.pack('!i', -71)
        sent = sock.sendto(network_message, (address, port))
