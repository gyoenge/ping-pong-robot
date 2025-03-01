import cv2
import numpy as np

# global variables
coordinate_list = []

# ==================================================================

def mouse_click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("클릭됨 ", end="// ")
        coordinate_list.append([x, y])

def approx_to_affinePar(approx):
    assert len(approx) == 4, "영역 꼭짓점은 4 점이어야 합니다. "
    ########## 우상, 좌상, 좌하, 우하 순으로 입력되었는지 확인 필요 ##########
    w = approx[3][0] - approx[2][0]
    h = max(approx[2][1] - approx[1][1], approx[3][1] - approx[0][1])
    pts1 = np.float32(approx)
    pts2 = np.float32([[w, 0], [0, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return matrix, w, h

# ==================================================================

cap1 = cv2.VideoCapture()
cap1.open(0 + cv2.CAP_DSHOW)  ########### 맞는 번호로 설정 ############
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
cap1.set(cv2.CAP_PROP_FOURCC, fourcc)

cv2.namedWindow('ClickCorners')
cv2.setMouseCallback('ClickCorners', mouse_click_event)

# ==================================================================

while cap1.isOpened():
    ret, frame1 = cap1.read()
    cv2.imshow('ClickCorners', frame1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("")
        aprrox1 = coordinate_list
        coordinate_list = []
        break
print("================ Clicked ROI Corners ===================")
print(f"aprrox1 = {aprrox1}")
print("========================================================")

while cap1.isOpened():
    ret, frame1 = cap1.read()
    matrix, w, h = approx_to_affinePar(aprrox1)
    ## ROI Result Show ##
    roi = cv2.warpPerspective(frame1, matrix, (w, h))
    cv2.imshow(f'frame', frame1)
    if roi is not None:
        cv2.imshow(f'ROI', roi)
    #####################
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("")
        break
print("============== ROI calibration parameters ================")
print(f"matrix1 = np.array({matrix.tolist()})")
print(f"w1 = {w}")
print(f"h1 = {h}")
print("==========================================================")

cv2.destroyAllWindows()
cap1.release()
