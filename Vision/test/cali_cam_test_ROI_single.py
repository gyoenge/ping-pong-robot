import cv2
import numpy as np

if __name__ == "__main__":
    ################## gain_ROIcalPar.py 에서 얻은 parameter 값들 입력 ##################
    matrix1 = np.array([[1.0748153809740302, -0.006824224641104997, -76.92266015453504],
                        [0.03520215969632352, 1.0736658707378657, -75.54383470831016],
                        [9.408801834129276e-05, 5.5429354776188645e-05, 1.0]])
    w1 = 497
    h1 = 329
    ###################################################################################

    cap = cv2.VideoCapture()
    cap.open(0 + cv2.CAP_DSHOW)
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    while True:
        ret, frame = cap.read()
        ROI = cv2.warpPerspective(frame, matrix1, (w1, h1))
        cv2.imshow('Original Cam frame', frame)
        cv2.imshow('ROI calibrated Cam frame', ROI)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
