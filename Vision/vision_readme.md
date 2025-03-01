### Vision

- **vision_module_bundles.py**
  : 사용되는 vision 처리 작업들을 class들로 모듈화하여 모아놓은 python 코드입니다.

  - cam을 켜는 class openCam을 인스턴스화하고, 해당 객체의 selp.cap을 다른 작업 클래스들의 초기화 인자로 할당하여 사용합니다.
  - 주요 작업 클래스들 및 각각의 설명은 다음과 같습니다.

    - ROIcalibration_makePar_single
      : single webcam 입력으로부터 탁구대 영역 ROI 추출을 위한 calibration parameter(matrix,w,h)를 구하여 'ROIcaliPar_single.txt'파일에 저장합니다.
    - ROIcalibration_single
      : 'ROIcaliPar_single.txt'파일의 calibration parameter를 불러온 후, single webcam 입력으로부터 탁구대 영역 ROI 추출 calibration을 수행합니다.
    - STEREOcalibration_makPar
      : stereo webcam 입력으로부터 stereo vision 수행을 위한 stereo calibration parameter를 구하여 'stereoMap.xml'파일에 저장합니다.
    - STEREOcalibration
      : 'stereoMap.xml'파일의 calibration parameter를 불러온 후, stereo webcam 입력으로부터 stereo calibration을 수행합니다.
    - calDepth
      : stereo vision triangulation을 사용하여, disparity(cam1, cam2의 각 프레임간의 x값의 차이)로부터 zDepth(카메라로부터 물체까지의 z축 수직거리)을 계산합니다.
    - calBallPos3D
      : stereo vision triangulation을 사용하여 zDepth값을 이용해 탁구공의 z 좌표값을 구한 후, 이를 이용해 보정된 탁구공 x, y 좌표값을 구하고, 최종적으로 탁구공의 3차원 위치 좌표 (x,y,z)를 반환합니다.
    - lineMethod_2Frame
      : 탁구공이 출발하는 첫 2프레임의 (x,y)값을 이용해, 탁구공의 전체 궤적을 xy평면상의 직선으로 근사하고, 최종적으로 Linear Actuator가 이동해야 하는 LX값을 반환합니다.

<br>

- **vision_module_bundles_TEST.ipynb**
  : 'vision_module_bundles.py' 모듈들이 잘 작동하는지 테스트하는 코드입니다.
