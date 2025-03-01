import cv2
import numpy as np
import multiprocessing as mp
import socket
import struct 
import time

sync_event = mp.Event()

def capture_left_frame(queue, event):
    cap = cv2.VideoCapture()
    cap.open(0 + cv2.CAP_DSHOW)
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    while True:
        ret, frame = cap.read()
        queue.put(frame)
        event.set()  # 이벤트 설정
    cap.release()

def capture_right_frame(queue, event):
    cap = cv2.VideoCapture()
    cap.open(1 + cv2.CAP_DSHOW)
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    while True:
        event.wait()  # 대기
        ret, frame = cap.read()
        event.clear()  # 이벤트 초기화
        if not ret:
            break
        queue.put(frame)
    cap.release()
    

# 받아온 프레임으로부터 mask 받기
def capture_left_frame_next(queue, queue_out) :
    matrix = np.array([[1.0580739547356008, -0.003245625628023323, -46.44490273701364], [0.02523642643295756, 1.0578268746481418, -37.07651650108695], [9.820236833344102e-05, 1.1090506116598113e-05, 1.0]])
    w = 505
    h = 342
    framew = 505 
    frameh = 342
    level = 0
    lower_orange = np.array([0, 100, 100])
    upper_orange = np.array([30, 255, 255])
    while True :
        frame = queue.get()
        frame = cv2.warpPerspective(frame, matrix, (w, h))
        frame = cv2.resize(frame, (framew, frameh))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        cv2.imshow('mask_left', mask)
        countN = cv2.countNonZero(mask)
        if countN > 0 :
            queue_out.put(mask)
        else :
            queue_out.put(None)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

def capture_right_frame_next(queue, queue_out) :
    matrix = np.array([[1.04433923044541, -0.006406989143836863, -116.78659811385847], [0.012479316070617709, 1.0441027779083543, -30.632561181343107], [6.161827908682351e-05, 1.7169740954759853e-05, 1.0]])
    w = 505
    h = 336
    framew = 505 
    frameh = 342
    level = 0
    lower_orange = np.array([0, 110, 110])
    upper_orange = np.array([30, 255, 255])
    while True :
        frame = queue.get()
        frame = cv2.warpPerspective(frame, matrix, (w, h))
        frame = cv2.resize(frame, (framew, frameh))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        cv2.imshow('mask_right', mask)
        countN = cv2.countNonZero(mask)
        if countN > 0 :
            queue_out.put(mask)
        else :
            queue_out.put(None)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

# mask로부터 무게중심 계산하기
def print_center_left(queue_left, queue_right, queue_out):
    while True:
        mask_left = queue_left.get()
        mask_right = queue_right.get()
        if mask_left is not None and mask_right is not None :
            M = cv2.moments(mask_left)
            cX_left = int(M["m10"] / M["m00"])
            cY_left = int(M["m01"] / M["m00"])
            M = cv2.moments(mask_right)
            cX_right = int(M["m10"] / M["m00"])
            cY_right = int(M["m01"] / M["m00"])
            queue_out.put([(cX_left, cY_left), (cX_right, cY_right)])

# 진짜 좌표 계산하기
def calculate_active_pos(queue, queue_out) :
    rx_coefficients = [-1.8080768996269398, 0.0, -1.9153036392626783, 2.9579464683982266, -0.06600191389658591, -0.005762039185511809, 0.02139037582654558, -0.00040367212441286103, -0.015794932883525875, 0.0004407141963407577, 0.0005120661318366041, 3.728496526811097e-05, -6.430319119550458e-05, -2.3131722873712678e-06, 1.713116034447859e-05, 3.091670806088116e-06, 1.3375711881852527e-06, 1.0108829051689804e-05, -6.962338038318805e-07, -1.7852398572215655e-06, -8.749896239779176e-07]
    ry_coefficients = [-12.313320933074465, 0.0, -2.2667179320538797, 2.392790990799181, 1.0252015666940442, -0.009865625406764146, 0.02013243475347843, 0.009673973654669127, -0.01067528024743924, -0.009988490838541439, 0.00021719843249660048, 1.986317791744312e-05, -6.0832210325508674e-05, 5.3357705126599976e-05, 6.11876710224762e-05, -0.00010582193392572087, 1.0227507692591505e-07, -1.982980840465515e-05, 5.285850928560265e-05, 3.3762349266330816e-07, -7.704657689622252e-07]
    #framew = 505
    #frameh = 342
    #frame = np.ones((frameh, framew, 3), np.uint8) * 255
    def polynomial_features(x, y, z):
        return [
            1, x, y, z, 
            x**2, x*y, x*z, y**2, y*z, z**2, 
            x**3, x**2*y, x**2*z, x*y**2, x*y*z, x*z**2, 
            y**3, y**2*z, y*z**2, z**3
        ]
    while True :
        getp = queue.get()
        pos_left = getp[0]
        pos_right = getp[1]
        if pos_left is None or pos_right is None :
            queue_out.put(None)
            continue
        else :
            features = polynomial_features(pos_left[0], pos_right[0], pos_left[1])
            rx_prediction = rx_coefficients[0] + sum([coef * feat for coef, feat in zip(rx_coefficients[1:], features)])
            ry_prediction = ry_coefficients[0] + sum([coef * feat for coef, feat in zip(ry_coefficients[1:], features)])
            queue_out.put((rx_prediction, ry_prediction))
            
            #rx_int = int(min(max(0, rx_prediction), framew - 1))
            #ry_int = int(min(max(0, ry_prediction), frameh - 1))
            #cv2.circle(frame, (rx_int, ry_int), 5, (0, 0, 255), -1) 
            #rx_int = int(min(max(0, pos_left[0]), framew - 1))
            #ry_int = int(min(max(0, pos_left[1]), frameh - 1))
            #cv2.circle(frame, (rx_int, ry_int), 5, (255, 0, 0), -1) 
            #cv2.imshow('Predicted Position', frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    frame = np.ones((frameh, framew, 3), np.uint8) * 255
            #    cv2.destroyAllWindows()
            
# 직선의 방정식에 이용할 좌표 추정하기
def estimate_line(queue, queue1, queue2, queue3, queue4, queue5) :
    pre_pos = None
    y_len = 262.3*(480/137)
    address = '192.168.0.106'
    port = 12345
    max_pos = 141
    curr_pos = 71
    message = 71
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    network_message = struct.pack('!i', message)
    sent = sock.sendto(network_message, (address, port))
    time_limit = 2
    while True :
        pos = queue.get()
        if pos is None :
            pre_pos = None
        if pre_pos is not None and pos is not None :
            poss = [pre_pos, pos]
            
            print('1st pos : ', pre_pos)
            print('2nd pos : ', pos)
            pre_pos = pos
            if (poss[1][1] > poss[0][1]) :
                LX = poss[0][0] + ((poss[1][0]-poss[0][0])/(poss[1][1]-poss[0][1]))*(y_len-poss[0][1])
                mm = (141-141*LX/640)
                print(mm)
                message = int(mm - curr_pos)

                if curr_pos + message > max_pos:
                    message = max_pos - curr_pos
                elif curr_pos + message < 0:
                    message = -curr_pos
                try:
                    #print(message)
                    network_message = struct.pack('!i', message)
                    sent = sock.sendto(network_message, (address, port))
                    curr_pos += message
                    time.sleep(3)
                    
                    network_message = struct.pack('!i', -message)
                    sent = sock.sendto(network_message, (address, port))
                    curr_pos -= message
                    print('now pos : ', curr_pos)
                except:
                    print("Linear-control-error")
                time.sleep(time_limit)
                for i in range(30) :
                    try :
                        queue.get(block=False)
                        queue1.get(block=False)
                        queue2.get(block=False)
                        queue3.get(block=False)
                        queue4.get(block=False)
                        queue5.get(block=False)
                        pre_pos = None
                    except :
                        pre_pos = None
        else :
            pre_pos = pos
        
if __name__ == "__main__":
    queue_len = 2
    queue_left_frame = mp.Queue(maxsize=queue_len)
    queue_right_frame = mp.Queue(maxsize=queue_len)
    queue_left_mask = mp.Queue(maxsize=queue_len)
    queue_right_mask = mp.Queue(maxsize=queue_len)
    queue_left_center = mp.Queue(maxsize=queue_len)
    queue_active_pos = mp.Queue(maxsize=queue_len)

    process1 = mp.Process(target=capture_left_frame, args=(queue_left_frame, sync_event,))
    process2 = mp.Process(target=capture_right_frame, args=(queue_right_frame, sync_event,))
    process3 = mp.Process(target=capture_left_frame_next, args=(queue_left_frame, queue_left_mask, ))
    process4 = mp.Process(target=capture_right_frame_next, args=(queue_right_frame, queue_right_mask, ))
    process5 = mp.Process(target=print_center_left, args=(queue_left_mask, queue_right_mask, queue_left_center, ))
    process6 = mp.Process(target=calculate_active_pos, args=(queue_left_center, queue_active_pos, ))
    process7 = mp.Process(target=estimate_line, args=(queue_active_pos, queue_left_frame, queue_right_frame, queue_left_mask, queue_right_mask, queue_left_center, ))

    try :
        process1.start()
        process2.start()
        process3.start()
        process4.start()
        process5.start()
        process6.start()
        process7.start()

        process1.join()
        process2.join()
        process3.join()
        process4.join()
        process5.join()
        process6.join()
        process7.join()
    except :
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        address = '192.168.0.106'
        port = 12345
        network_message = struct.pack('!i', -71)
        sent = sock.sendto(network_message, (address, port))