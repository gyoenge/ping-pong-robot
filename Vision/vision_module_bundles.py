import cv2
import numpy as np

import threading
import time

import glob 
import os
import pickle

""" ############################################################# 
    cam을 키는 클래스 
    : 아래 작업 클래스를 수행하기 전 반드시 사전 수행되어야 함 
    ############################################################# """

# cam 키기 
class openCam: 
    def __init__(self, camID, is_wide=True): 
        self.cap = cv2.VideoCapture()
        self.cap.open(camID + cv2.CAP_DSHOW)  
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc) 
        if is_wide:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360) 
        else: 
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 
            
    def getFrameSize(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (width, height)
        return frame_size

    def release(self): 
        self.cap.release() 

class CameraThread(threading.Thread): # threading용 
    def __init__(self, camID, is_wide=True):
        threading.Thread.__init__(self)
        self.camID = camID

        self.cap = cv2.VideoCapture()
        self.cap.open(camID + cv2.CAP_DSHOW)  
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc) 
        if is_wide:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360) 
        else: 
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 

        self.current_frame = None

    def run(self):
        while True:
            ret, self.current_frame = self.cap.read()

""" ############################################################# 
    cam을 사용하는 작업 클래스들 
    : 모든 close에는 self.cap.release()를 포함할 것 
    ############################################################# """

### calibrations ##########################################################################################

# ROI calibaration 수행에 필요한 parameter 파일을 만들어서 저장하기 (for single cam)
class ROIcalibration_makePar_single: 
    def __init__(self, cap): 
        self.cap = cap 
        self.corner_list = [] 
        
    def mouse_click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("클릭됨 ", end="// ")
            self.corner_list.append([x, y])

    def approx_to_affinePar(self, approx):
        assert len(approx) == 4, "영역 꼭짓점은 4 점이어야 합니다. "
        ########## 우상, 좌상, 좌하, 우하 순으로 입력되었는지 확인 필요 ##########
        w = approx[3][0] - approx[2][0]
        h = max(approx[2][1] - approx[1][1], approx[3][1] - approx[0][1])
        pts1 = np.float32(approx)
        pts2 = np.float32([[w, 0], [0, 0], [0, h], [w, h]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        return matrix, w, h
        
    def createCalibMap(self): 
        cv2.namedWindow('ClickCorners') 
        cv2.setMouseCallback('ClickCorners', self.mouse_click_event)
        
        while self.cap.isOpened(): 
            ret, frame = self.cap.read() 
            cv2.imshow('ClickCorners', frame) 
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("")
                print(f"clicked corners (x,y) : {self.corner_list}")
                break
                
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            matrix, w, h = self.approx_to_affinePar(self.corner_list)
            ## ROI Result Show ##
            roi = cv2.warpPerspective(frame, matrix, (w, h))
            cv2.imshow(f'frame', frame)
            if roi is not None:
                cv2.imshow(f'ROI', roi)
            #####################
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("")
                break
                
        print("============== ROI calibration parameters ================")
        print(f"matrix = np.array({matrix.tolist()})")
        print(f"w = {w}")
        print(f"h = {h}")
        print("==========================================================")
        
        calibration_data = {'matrix': matrix, 'w': w, 'h': h}
        with open('ROIcaliPar_single.txt', 'wb') as f:
            pickle.dump(calibration_data, f)
        print("Saved calibration data to 'ROIcaliPar_single.txt'")
        
    def close(self): 
        self.cap.release() 
        cv2.destroyAllWindows() 

# ROI calibaration 수행하기 (for single cam)
class ROIcalibration_single: 
    def __init__(self, cap): 
        self.cap = cap
        with open('ROIcaliPar_single.txt', 'rb') as f:
            calibration_data = pickle.load(f)
            self.matrix = calibration_data['matrix']
            self.w = calibration_data['w']
            self.h = calibration_data['h']
            
    def ROIrectify(self, frame): 
        roi = cv2.warpPerspective(frame, self.matrix, (self.w, self.h))
        return roi
        
    def start(self): 
        while self.cap.isOpened():    
            succes, frame = self.cap.read()

            roi = self.ROIrectify(frame) 

            cv2.imshow("frame", frame) 
            cv2.imshow("roi", roi) 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()   

# ROI calibaration IMG TEST (for single cam, image test)
class ROIcalibration_single_helpers: 
    def __init__(self):
        self.corner_list = [] 
        self.matrix
        self.w
        self.h

    def mouse_click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("클릭됨 ", end="// ")
            self.corner_list.append([x, y]) 

    def approx_to_affinePar(self, approx):
        assert len(approx) == 4, "영역 꼭짓점은 4 점이어야 합니다. "
        ########## 우상, 좌상, 좌하, 우하 순으로 입력되었는지 확인 필요 ##########
        w = approx[3][0] - approx[2][0]
        h = max(approx[2][1] - approx[1][1], approx[3][1] - approx[0][1])
        pts1 = np.float32(approx)
        pts2 = np.float32([[w, 0], [0, 0], [0, h], [w, h]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        return matrix, w, h

    def createROICalibMap(self, frame): 
        cv2.namedWindow('ClickCorners') 
        cv2.setMouseCallback('ClickCorners', self.mouse_click_event)
        
        cv2.imshow('ClickCorners', frame) 
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("")
            print(f"clicked corners (x,y) : {self.corner_list}")
                
        self.matrix, self.w, self.h = self.approx_to_affinePar(self.corner_list)
        ## ROI Result Show ##
        roi = cv2.warpPerspective(frame, self.matrix, (self.w, self.h))
        cv2.imshow(f'frame', frame)
        if roi is not None:
            cv2.imshow(f'ROI', roi)
        #####################
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("") 
                
        print("============== ROI calibration parameters ================")
        print(f"matrix = np.array({self.matrix.tolist()})")
        print(f"w = {self.w}")
        print(f"h = {self.h}")
        print("==========================================================")
    
    def ROIrectify(self, frame): 
        roi = cv2.warpPerspective(frame, self.matrix, (self.w, self.h))
        return roi

# ROI calibaration 수행에 필요한 parameter 파일을 만들어서 저장하기 (for stereo cam)
class ROIcalibration_makePar_stereo: 
    def __init__(self, cap1, cap2): 
        self.cap1 = cap1 
        self.cap2 = cap2 
        self.corner_list1 = [] 
        self.corner_list2 = [] 
        
    def mouse_click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("클릭됨 ", end="// ")
            self.corner_list.append([x, y])

    def approx_to_affinePar(self, approx):
        assert len(approx) == 4, "영역 꼭짓점은 4 점이어야 합니다. "
        ########## 우상, 좌상, 좌하, 우하 순으로 입력되었는지 확인 필요 ##########
        w = approx[3][0] - approx[2][0]
        h = max(approx[2][1] - approx[1][1], approx[3][1] - approx[0][1])
        pts1 = np.float32(approx)
        pts2 = np.float32([[w, 0], [0, 0], [0, h], [w, h]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        return matrix, w, h
        
    def createCalibMap(self): 
        cv2.namedWindow('ClickCorners') 
        cv2.setMouseCallback('ClickCorners', self.mouse_click_event)
        
        while self.cap1.isOpened(): 
            ret1, frame1 = self.cap1.read() 
            cv2.imshow('ClickCorners', frame1) 
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("")
                print(f"clicked corners (x,y) in cam1 : {self.corner_list1}")
                break
                
        while self.cap1.isOpened():
            ret1, frame1 = self.cap1.read() 
            matrix1, w1, h1 = self.approx_to_affinePar(self.corner_list1)
            ## ROI Result Show ##
            roi1 = cv2.warpPerspective(frame1, matrix1, (w1, h1))
            cv2.imshow(f'frame1', frame1)
            if roi1 is not None:
                cv2.imshow(f'ROI1', roi1)
            #####################
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("")
                break

        while self.cap2.isOpened(): 
            ret2, frame2 = self.cap2.read() 
            cv2.imshow('ClickCorners', frame2) 
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("")
                print(f"clicked corners (x,y) in cam2 : {self.corner_list2}")
                break
                
        while self.cap2.isOpened():
            ret2, frame2 = self.cap2.read() 
            matrix2, w2, h2 = self.approx_to_affinePar(self.corner_list2)
            ## ROI Result Show ##
            roi2 = cv2.warpPerspective(frame2, matrix2, (w2, h2))
            cv2.imshow(f'frame2', frame2)
            if roi2 is not None:
                cv2.imshow(f'ROI2', roi2)
            #####################
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("")
                break
                
        print("============== ROI calibration parameters ================")
        print(f"matrix1 = np.array({matrix1.tolist()})")
        print(f"w1 = {w1}")
        print(f"h1 = {h1}")
        print(f"matrix2 = np.array({matrix2.tolist()})")
        print(f"w2 = {w2}")
        print(f"h2 = {h2}")
        print("==========================================================")
        
        calibration_data = {'matrix1': matrix1, 'w1': w1, 'h1': h1, 'matrix2': matrix2, 'w2': w2, 'h2': h2}
        with open('ROIcaliPar_stereo.txt', 'wb') as f:
            pickle.dump(calibration_data, f)
        print("Saved calibration data to 'ROIcaliPar_stereo.txt'")
        
    def close(self): 
        self.cap1.release() 
        self.cap2.release() 
        cv2.destroyAllWindows() 

# ROI calibaration 수행하기 (for stereo cam)
class ROIcalibration_stereo: 
    def __init__(self, cap1, cap2): 
        self.cap1 = cap1
        self.cap2 = cap2
        with open('ROIcaliPar_stereo.txt', 'rb') as f:
            calibration_data = pickle.load(f)
            self.matrix1 = calibration_data['matrix1']
            self.w1 = calibration_data['w1']
            self.h1 = calibration_data['h1']
            self.matrix2 = calibration_data['matrix2']
            self.w2 = calibration_data['w2']
            self.h2 = calibration_data['h2']
            
    def ROIrectify(self, frame, matrix, w, h): 
        roi = cv2.warpPerspective(frame, matrix, (w, h))
        return roi
        
    def start(self): 
        while self.cap1.isOpened() and self.cap2.isOpened():    
            succes1, frame1 = self.cap1.read()
            succes2, frame2 = self.cap2.read()

            roi1 = self.ROIrectify(frame1, self.matrix1, self.w1, self.h1) 
            roi2 = self.ROIrectify(frame2, self.matrix2, self.w2, self.h2)

            cv2.imshow("frame1", frame1) 
            cv2.imshow("roi1", roi1) 
            cv2.imshow("frame2", frame2) 
            cv2.imshow("roi2", roi2) 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    def close(self):
        self.cap1.release() 
        self.cap2.release() 
        cv2.destroyAllWindows()   

# stereo calibaration 수행에 필요한 parameter 파일을 만들어서 저장하기
class STEREOcalibration_makPar: 
    def __init__(self, cap1, cap2, calibImg_rootpath='./images'): 
        self.cap1 = cap1
        self.cap2 = cap2
        self.left_path = f'{calibImg_rootpath}/calibrationStereo/stereoLeft/' 
        self.right_path = f'{calibImg_rootpath}/calibrationStereo/stereoRight/' 
        
    def captureCalibImg(self, will_resetImg=True): # : checkboard calib image capture 
        num = 0
        while self.cap1.isOpened() and self.cap2.isOpened():
            succes1, img1 = self.cap1.read()
            succes2, img2 = self.cap2.read()

            if will_resetImg: # img 저장 폴더 초기화 
                if os.path.exists(self.left_path):
                    for filename in os.listdir(self.left_path):
                        os.remove(self.left_path + filename)
                if os.path.exists(self.right_path):
                    for filename in os.listdir(self.right_path):
                        os.remove(self.right_path + filename)
            
            if not os.path.exists(self.left_path): # img 저장 경로 없으면 생성 
                os.makedirs(self.left_path)
            if not os.path.exists(self.right_path):
                os.makedirs(self.right_path)

            k = cv2.waitKey(5)
            if k == ord('s'): # wait for 's' key to save and exit 
                cv2.imwrite(self.left_path + f'calibL{num}.png', img1) 
                cv2.imwrite(self.right_path + f'calibR{num}.png', img2) 
                print(f"images no.{num} saved!") 
                num += 1

            cv2.imshow('CAM 1', img1)
            cv2.imshow('CAM 2', img2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    def improveCalibImg(self, imgname='calib'): # : calib img quality 개선 
        imagesLeft = glob.glob(self.left_path + imgname + '*.png')
        imagesRight = glob.glob(self.right_path + imgname + '*.png')
        
        def improve(img):
            # Histogram Equalization
#             img = cv2.equalizeHist(img) 
            # Adaptive Histogram Equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img)
            # Unsharp Masking 
            gaussian = cv2.GaussianBlur(img, (5,5), 5.0) # 커널크기 & 표준편차 조절 
            img = cv2.addWeighted(img, 2.0, gaussian, -0.5, 0, img) 
            # 밝기 증가
            img = cv2.convertScaleAbs(img, alpha=1, beta=-20) # 대비 & 밝기 조절 
            
            return img
        
        for imgLeft, imgRight in zip(imagesLeft, imagesRight):
            print(".")
            imgL = cv2.imread(imgLeft)
            imgR = cv2.imread(imgRight)
            improvedL = improve(imgL)
            improvedR = improve(imgR)
            cv2.imwrite(f'{self.left_path}improved_{os.path.basename(imgLeft)}', improvedL)
            cv2.imwrite(f'{self.left_path}improved_{os.path.basename(imgRight)}', improvedR)
        
    
    def createCalibMap(self, chessboardSize=(7,6)): # : save calibration parameters to stereoMap.xml
        ############# FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############
        
        frameSize = self.cap1.getFrameSize()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)
        objpoints = [] # 3d point in real world space
        imgpointsL = [] # 2d points in image plane.
        imgpointsR = [] # 2d points in image plane.
        
        imagesLeft = glob.glob(self.left_path + '*.png')
        imagesRight = glob.glob(self.right_path + '*.png')
        
        for imgLeft, imgRight in zip(imagesLeft, imagesRight):
            print(".")
            imgL = cv2.imread(imgLeft)
            imgR = cv2.imread(imgRight)
            grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
            cv2.imshow('grayL', grayL)
            cv2.imshow('grayR', grayR)

            retL, cornersL = cv2.findChessboardCorners(grayL, chessboardSize, None)
            retR, cornersR = cv2.findChessboardCorners(grayR, chessboardSize, None)

            # ============================= test code =============================
            if retL and retR:
                print(f"Corners were found in the image: {imgLeft}")
            else:
                print(f"Corners were not found in the image: {imgLeft}")
            # =====================================================================
            
            if retL and retR == True:
                objpoints.append(objp)

                cornersL = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
                imgpointsL.append(cornersL)
                cornersR = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
                imgpointsR.append(cornersR)

                cv2.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
                cv2.imshow('img left', imgL)
                cv2.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
                cv2.imshow('img right', imgR)
                cv2.waitKey(1000)
                
        cv2.destroyAllWindows()
        
        ############## CALIBRATION #######################################################
        
        retL, cameraMatrixL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
        heightL, widthL, channelsL = imgL.shape
        newCameraMatrixL, roi_L = cv2.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

        retR, cameraMatrixR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
        heightR, widthR, channelsR = imgR.shape
        newCameraMatrixR, roi_R = cv2.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

        ########## Stereo Vision Calibration #############################################

        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC

        criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)

        #print(newCameraMatrixL)
        #print(newCameraMatrixR)

        ########## Stereo Rectification #################################################

        rectifyScale = 1
        rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv2.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale,(0,0))

        stereoMapL = cv2.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv2.CV_16SC2)
        stereoMapR = cv2.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv2.CV_16SC2)

        print("Saving parameters!")
        cv_file = cv2.FileStorage('stereoMap.xml', cv2.FILE_STORAGE_WRITE)

        cv_file.write('stereoMapL_x',stereoMapL[0])
        cv_file.write('stereoMapL_y',stereoMapL[1])
        cv_file.write('stereoMapR_x',stereoMapR[0])
        cv_file.write('stereoMapR_y',stereoMapR[1])

        cv_file.release()
            
    def close(self): 
        self.cap1.release()
        self.cap2.release() 
        cv2.destroyAllWindows()  

# stereo calibaration 수행하기 
class STEREOcalibration:  # 수행하기 전 calibImg로 생성된 stereoMap.xml를 만들어야 함
    def __init__(self, cap1, cap2): 
        self.cap1 = cap1
        self.cap2 = cap2
        
        cv_file = cv2.FileStorage()
        cv_file.open('stereoMap.xml', cv2.FileStorage_READ)
        self.stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
        self.stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
        self.stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
        self.stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
        
    def undistortRectify(self, frameL, frameR): # : stereoMap.xml로 calibration하는 helper 
        undistortedL= cv2.remap(frameL, self.stereoMapL_x, self.stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        undistortedR= cv2.remap(frameR, self.stereoMapR_x, self.stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        return undistortedL, undistortedR
    
    def start(self): # : frame1, frame2에 대해 stereo calibration 수행 
        while(self.cap1.isOpened() and self.cap2.isOpened()):    
            succes1, frame1 = self.cap1.read()
            succes2, frame2 = self.cap2.read()

            undistorted1, undistorted2 = self.undistortRectify(frame1, frame2) 

            cv2.imshow("frame1", frame1) 
            cv2.imshow("frame2", frame2)
            cv2.imshow("undistorted1", undistorted1) 
            cv2.imshow("undistorted2", undistorted2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def close(self):
        self.cap1.release()
        self.cap2.release() 
        cv2.destroyAllWindows()  

### 부수 작업들 ##########################################################################################

# image에서 위치 찾기 
class clickPointPos: 
    def __init__(self, cap): 
        self.cap = cap
        self.coordinate_list = []
        
    def mouse_click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("클릭됨 ", end="// ")
            self.coordinate_list.append([x, y])
    
    def start(self): 
        cv2.namedWindow('ClickPoints') 
        cv2.setMouseCallback('ClickPoints', self.mouse_click_event)
        
        while self.cap.isOpened(): 
            ret, frame = self.cap.read() 
            cv2.imshow('ClickPoints', frame) 
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("")
                print(f"clicked points (x,y) : {self.coordinate_list}")
                break

    def close(self):
        self.cap.release()       
        cv2.destroyAllWindows()  

# ROI frame에서 orange ball detecting을 하기 위한 helper 함수들 모음 
class orangeBall_helpers:
    LOWER_ORANGE = (0,50,50)
    UPPER_ORANGE = (30,255,255)
    Y_inrange_MIN_default = 0
    Y_inrange_MAX_default = 160

    def getOrangeMask(frame, lower_orange=LOWER_ORANGE, upper_orange=UPPER_ORANGE):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        orange_mask = cv2.inRange(hsv, np.array(lower_orange), np.array(upper_orange))
        orange_object = cv2.bitwise_and(frame, frame, mask=orange_mask) 
        return orange_mask, orange_object

    def checkOrangeInRange(orange_mask, ymin=Y_inrange_MIN_default, ymax=Y_inrange_MAX_default):
        orange_pixels = cv2.findNonZero(orange_mask)
        if orange_pixels is None:
            return False
        for pixel in orange_pixels:
            y = pixel[0][1]
            if ymin <= y <= ymax:
                return True
        return False

    def detectBallPos(orange_mask):
        contours, hierarchy = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 1:
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    #             print(f"[cam1] | ({cX}, {cY})")
                return (cX, cY)
        elif len(contours) >= 1:
            max_area = 0
            largest_contour = None
            for c in contours:
                area = cv2.contourArea(c)
                if area > max_area:
                    max_area = area
                    largest_contour = c
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    #             print(f"[cam1] | ({cX}, {cY})")
                return (cX, cY)
            
    def detectBallContour(orange_mask):
        contours, hierarchy = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 1:
            return (contours[0])
        elif len(contours) >= 1:
            max_area = 0
            largest_contour = None
            for c in contours:
                area = cv2.contourArea(c)
                if area > max_area:
                    max_area = area
                    largest_contour = c
            return largest_contour


# save capture frames (for single cam, raw frame)
class saveCapFrames_single: 
    def __init__(self, cap, path=""): 
        self.cap = cap
        self.path = path

    def start(self): 
        num = 0
        while self.cap.isOpened():
            succes1, img = self.cap.read()
            k = cv2.waitKey(5)
            if k == ord('s'): # wait for 's' key to save and exit
                cv2.imwrite(self.path + '/single' + str(num).zfill(2) + '.png', img) # 이미지 저장 경로 설정 
                print("images saved!")
                num += 1
            cv2.imshow('CAM',img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    def close(self):
        self.cap.release()
        cv2.destroyAllWindows() 

# save capture video (for single cam, raw frame)
class saveCapVideo_single: 
    def __init__(self, cap, path=""): 
        self.cap = cap
        self.path = path

    def start(self): 
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.path + '/output.avi', fourcc, 30.0, (frame_width,frame_height))

        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret == True:
                out.write(frame)
                cv2.imshow('Frame', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
    
    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()   

# save capture frames (for stereo cam)
class saveCapFrames_stereo: 
    def __init__(self, cam1ID, cam2ID, path=""): 
        self.cam1 = CameraThread(cam1ID) # 동시에 프레임 가져옴 
        self.cam2 = CameraThread(cam2ID) 
        self.cam1.start()
        self.cam2.start()
        
        self.path = path

        self.frame_counter1 = 0
        self.frame_counter2 = 0

    def print_timer(self):
            print("10초 경과")
            timer = threading.Timer(10, self.print_timer) # 다음 타이머 설정
            timer.start()    

    def refresh_frameCounter(self, init_num=0):
        self.frame_counter1 = init_num
        self.frame_counter2 = init_num
    
    def start(self): 
        save_frames = False
        self.print_timer() # 타이머 시작

        while True:
            frame1 = self.cam1.current_frame
            frame2 = self.cam2.current_frame

            if frame1 is not None:
                cv2.imshow('Camera 1', frame1)
                if save_frames:
                    cv2.imwrite(self.path + f'/cam1_{self.frame_counter1}.jpg', frame1)
                    print(f'cam1/{self.frame_counter1}.jpg saved')
                    self.frame_counter1 += 1

            if frame1 is not None:
                cv2.imshow('Camera 2', frame2)
                if save_frames:
                    cv2.imwrite(self.path + f'/cam2_{self.frame_counter2}.jpg', frame2)
                    print(f'cam2/{self.frame_counter2}.jpg saved')
                    self.frame_counter2 += 1

            key = cv2.waitKey(1) & 0xFF
            # 's'키를 누르면 프레임 저장을 시작합니다
            if key == ord('s'):
                save_frames = True
            # 'q'키를 누르면 루프를 종료합니다
            elif key == ord('q'):
                break
        
    def close(self):
        self.cam1.join()
        self.cam2.join()
        cv2.destroyAllWindows()  

# save capture video (for stereo cam)
# class saveCapVideo_stereo: 

### 주요 작업들 ##########################################################################################

# depth, ballPos3D constants 
BASELINE = 10.5 
ALPHA = 71 
CAMSTOTABLE = 157 

# depth 계산
class calDepth:
    def __init__(self, cap1, cap2, baseline = BASELINE, alpha = ALPHA): 
        self.cap1 = cap1
        self.cap2 = cap2
        
#         self.frame_rate = 23 # Camera frame rate (maximum at 120 fps)
        self.baseline = baseline    # Distance between the cameras [cm]
        self.f = 10                 # Camera lense's focal length [mm]
        self.alpha = alpha          # Camera field of view in the horisontal plane [degrees]
        
    def find_depth(self, right_point, left_point, frame_right, frame_left): # : find depth using triangulation 
        # CONVERT FOCAL LENGTH f FROM [mm] TO [pixel]:
        height_right, width_right, depth_right = frame_right.shape
        height_left, width_left, depth_left = frame_left.shape
        if width_right == width_left:
            f_pixel = (width_right * 0.5) / np.tan(self.alpha * 0.5 * np.pi/180)
        else:
            print('Left and right camera frames do not have the same pixel width')

        x_right = right_point[0]
        x_left = left_point[0]
        # CALCULATE THE DISPARITY:
        disparity = x_left-x_right      #Displacement between left and right frames [pixels]
        # CALCULATE DEPTH z:
        zDepth = (self.baseline*f_pixel)/disparity             #Depth in [cm]

        return zDepth

    def start(self, left_point, right_point, is_point=True): 
        while(self.cap1.isOpened() and self.cap2.isOpened()):
            succes_right, frame_right = self.cap1.read()
            succes_left, frame_left = self.cap2.read()

            if not succes_right or not succes_left:                    
                break

            depth = self.find_depth(right_point, left_point, frame_right, frame_left) 
            depth = max(depth, -depth)
            cv2.putText(frame_right, "Distance: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
            cv2.putText(frame_left, "Distance: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)

            if is_point:
                cv2.circle(frame_right, left_point, 5, (0, 255, 0), -1)
                cv2.circle(frame_left, right_point, 5, (0, 255, 0), -1)
            
            cv2.imshow("frame right", frame_right) 
            cv2.imshow("frame left", frame_left) 

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Depth: ", str(round(depth,1)))
                break 
                
        cv2.destroyAllWindows()
            
    def close(self):
        self.cap1.release()
        self.cap2.release()

# 3차원 좌표 (x,y,z) 계산
class calBallPos3D:
    def __init__(self, cap1, cap2, camsToTable = CAMSTOTABLE, baseline = BASELINE, alpha = ALPHA): 
        self.calDepth = calDepth(cap1, cap2, baseline, alpha)
        self.camsToTable = camsToTable 

    def find_ballPos3D(self, left_point, frame_right, frame_left, zDepth): # : find (x,y,z) using triangulation 
        # CONVERT FOCAL LENGTH f FROM [mm] TO [pixel]:
        height_right, width_right, depth_right = frame_right.shape
        height_left, width_left, depth_left = frame_left.shape
        if width_right == width_left:
            f_pixel = (width_right * 0.5) / np.tan(self.calDepth.alpha * 0.5 * np.pi/180)
        else:
            print('Left and right camera frames do not have the same pixel width')

        x_left = left_point[0]
        y_left = left_point[1]
        # CALCULATE THE z pos:
        z = self.camsToTable - zDepth  # in [cm]
        # CALCULATE THE x, y pos:
        x = (zDepth*x_left)/f_pixel  # in [cm]
        y = (zDepth*y_left)/f_pixel  # in [cm]

        return x, y, z

    def start(self, left_point, right_point, is_point=True): 
        while(self.calDepth.cap1.isOpened() and self.calDepth.cap2.isOpened()):
            succes_right, frame_right = self.calDepth.cap1.read()
            succes_left, frame_left = self.calDepth.cap2.read()

            if not succes_right or not succes_left:                    
                break

            depth = self.calDepth.find_depth(right_point, left_point, frame_right, frame_left) 
            depth = max(depth, -depth)

            x, y, z = self.find_ballPos3D(left_point, frame_right, frame_left, depth)

            cv2.putText(frame_right, f"(x,y,z): ({x},{y},{z})", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
            cv2.putText(frame_left, f"(x,y,z): ({x},{y},{z})", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)

            if is_point:
                cv2.circle(frame_right, left_point, 5, (0, 255, 0), -1)
                cv2.circle(frame_left, right_point, 5, (0, 255, 0), -1)
            
            cv2.imshow("frame right", frame_right) 
            cv2.imshow("frame left", frame_left) 

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Depth: ", str(round(depth,1)))
                break 
                
        cv2.destroyAllWindows()
            
    def close(self):
        self.calDepth.cap1.release()
        self.calDepth.cap2.release()
        
# line method 로 LX 계산 (single cam, 2frame) 
class lineMethod_2Frame: 
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    TABLE_WIDTH = 137
    TABLE_HEIGHT = 262
    LIN_LENGTH = 141

    Y_LEN = TABLE_HEIGHT*(FRAME_HEIGHT/TABLE_WIDTH)  # table height (cm) 의 픽셀 단위 변환 

    def __init__(self, cap): 
        self.cap = cap

        # ROI par 불러오기 
        with open('ROIcaliPar_single.txt', 'rb') as f:
            calibration_data = pickle.load(f)
            self.matrix = calibration_data['matrix']
            self.w = calibration_data['w']
            self.h = calibration_data['h']
        
        # orange processing helper 함수들 
        self.orangeBall_helpers = orangeBall_helpers()

        # state 변수 
        self.frame_counter = 0
        self.dataColumn_counter = 0
        self.save_frames = False
        self.is_orange_in_range = False
        # 저장 변수 
        self.cam_points = [] 
        self.track_save = [] 

    def ROIrectify(self, frame): 
        roi = cv2.warpPerspective(frame, self.matrix, (self.w, self.h))
        return roi
        
    def calcul_linear_position(self, y_len, x1, y1, x2, y2) :
    #     if y2-y1 == 0:
    #         LX = Y_LEN  # 320 ?? 
    #     else: 
        LX = x1 + ((x2-x1)/(y2-y1))*(y_len-y1) # 픽셀 단위  (y_len도 픽셀 단위 )
        #if LX < 71 :
        #    LX *= 2 
        return (self.LIN_LENGTH/self.FRAME_WIDTH)*LX 
    
    def start(self): 
        while self.cap.isOpened(): 
            ret, frame = self.cap.read() 

            if frame is not None:
                roi = self.ROIrectify(frame) 
                orange_mask, orange_object = self.orangeBall_helpers.getOrangeMask(roi)
        #         cv2.imshow(f'ROI orange masking', orange_object)
                
                self.is_orange_in_range = self.orangeBall_helpers.checkOrangeInRange(orange_mask, 0, 160)
                if self.is_orange_in_range: 
                    self.save_frames = True 
                
                if self.save_frames:
                    ballPos = self.orangeBall_helpers.detectBallPos(orange_mask)
                    if ballPos not in self.cam_points and ballPos is not None:
                        self.cam_points.append(ballPos)
                    
        #             cv2.imwrite(f'images/ballPosDataCollect/col{self.dataColumn_counter}_cam_{self.frame_counter}.jpg', roi)
        #             print(f'col{self.dataColumn_counter}_cam_{self.frame_counter}.jpg saved')
                    self.frame_counter += 1 
                    
                    if not self.orangeBall_helpers.checkOrangeInRange(orange_mask, 0, 480): 
                        # 점 두개 -> 직선 식 
                        if len(self.cam_points) >= 2: 
                            temp = self.cam_points[0:2] 
                            if temp[1][1] != temp[0][1]: 
                                linPos = self.LIN_LENGTH - self.calcul_linear_position(self.Y_LEN, temp[0][0], temp[0][1], temp[1][0], temp[1][1])
                                print(f"두 점 좌표 : {temp} | 리니어 좌표 : {linPos}")
                                time.sleep(2) 
        #                     self.track_save.append([item for sublist in temp if sublist is not None for item in sublist])
        #                     print([item for sublist in temp if sublist is not None for item in sublist])
                        self.save_frames = False 
                        self.frame_counter = 0 
                        self.cam_points = [] 
                        self.dataColumn_counter += 1 

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("")
                break

    def close(self):
        self.cap.release()       
        cv2.destroyAllWindows()

# line method (single cam, 1frame, image test) 
class lineMethod_1Frame_imgTest: 
    def __init__(self, imgPath): 
        self.frame = cv2.imread(imgPath)

        # ROI calibration helper 함수들 
        self.ROIcalibration_single_helpers = ROIcalibration_single_helpers()

        # orange processing helper 함수들 
        self.orangeBall_helpers = orangeBall_helpers()

    def orangeCountour_to_linePar(c):  # c는 ball contour에 해당, contours 의 요소 하나 
        # find rotated bounding box
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # find min and max y-coordinate points
        topmost = tuple(c[c[:,:,1].argmin()][0])
        bottommost = tuple(c[c[:,:,1].argmax()][0])
        # find min and max x-coordinate points
        leftmost = tuple(c[c[:,:,0].argmin()][0])
        rightmost = tuple(c[c[:,:,0].argmax()][0])
        
        # 경로 case 분류
        if (topmost[0] > bottommost[0]):
            p1 = (topmost[0], rightmost[1])
            p2 = (bottommost[0], leftmost[1])
        else:
            p1 = (topmost[0], leftmost[1])
            p2 = (bottommost[0], rightmost[1])

        # calculate slope (m) and y-intercept (b) for the line equation: "y = mx + b"
        if p2[0] - p1[0] != 0:
            m = (p2[1] - p1[1]) / (p2[0] - p1[0])
            b = p1[1] - m * p1[0]
        else:
            # 기울기 수직 ------ 고치기 
            m = float('inf')
            b = 0 
            
        return m, b 

    def orangeCountour_to_lineDraw(c, roi):
        # find rotated bounding box
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # find min and max y-coordinate points
        topmost = tuple(c[c[:,:,1].argmin()][0])
        bottommost = tuple(c[c[:,:,1].argmax()][0])
        # find min and max x-coordinate points
        leftmost = tuple(c[c[:,:,0].argmin()][0])
        rightmost = tuple(c[c[:,:,0].argmax()][0])
        
        # 경로 case 분류
        if (topmost[0] > bottommost[0]):
            p1 = (topmost[0], rightmost[1])
            p2 = (bottommost[0], leftmost[1])
        else:
            p1 = (topmost[0], leftmost[1])
            p2 = (bottommost[0], rightmost[1])

        # draw
        cv2.line(roi, p1, p2, (0, 0, 255), 2)

        return roi
    
    def start(self): 
        # ROI calibration 
        self.ROIcalibration_single_helpers.createROICalibMap(self.frame)
        # get ROI 
        roi = self.ROIcalibration_single_helpers.ROIrectify(self.frame)
        
        # get orange ball contour
        orange_mask, orange_object = self.orangeBall_helpers.getOrangeMask(roi)
        ball_contour = self.orangeBall_helpers.detectBallContour(orange_mask)

        # get line parameters m and b 
        # m, b = self.orangeCountour_to_linePar(ball_contour)
        # print(f"RESULT line is y={int(m)}x+{int(b)}")

        # show RESULT 
        cv2.imshow(f'frame', self.frame)
        cv2.imshow(f'ROI', roi)
        # draw RESULT line in roi         
        linedRoi = self.orangeCountour_to_lineDraw(ball_contour, roi)
        cv2.imshow(f'ROI with RESULT line', linedRoi)
        
        #####################
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("")

    def close(self):     
        cv2.destroyAllWindows()

