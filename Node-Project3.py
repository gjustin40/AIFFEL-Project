import os

import dlib
import numpy as np
import cv2
import math

from PIL import Image

def cat_filter(image_path):
    image_path = image_path
    image = Image.open(image_path)

    image = np.array(image)
    image_show = image.copy() # 원본 이미지 저장

    detector_hog = dlib.get_frontal_face_detector() # 얼굴 탐지기 불러오기
    face_detection = detector_hog(image, 1)         # 얼굴 탐지 실시

    # 탐지된 얼굴 시각화
    # for rect in face_detection:
    #     left = rect.left()
    #     top = rect.top()
    #     right = rect.right()
    #     bottom = rect.bottom()

    #     cv2.rectangle(image, (left, top), (right, bottom), (0,255,0), 2) # 상자 그리기(얼굴 탐지)


    # 학습된 추출기의 feature 불러오기
    model_path = os.getenv('HOME')+'/aiffel/camera_sticker/models/shape_predictor_68_face_landmarks.dat'
    landmark_predictor = dlib.shape_predictor(model_path) # feature 적용

    landmark_image = image.copy()
    original_image = image_show.copy()
    points = landmark_predictor(image, face_detection[0]) # 탐지한 얼굴 부분을 활용해 특징점 추출
    landmarks = list(map(lambda p: (p.x, p.y), points.parts()))

    sticker_size = (int(rect.width() // 1.2), int(rect.height() // 1.2)) # 스티커 이미지 사이즈 조정

    sticker_path = os.getenv("HOME") + '/aiffel/camera_sticker/images/cat-whiskers.png'
    sticker = Image.open(sticker_path).convert('RGB')
    sticker_original = np.array(sticker.resize(sticker_size)).copy() # 비교를 위해 이미지 원본 저장

    # Padding 100 추가 / 빈 공간은 흰색(value=255)으로 설정 / value 값으로 동일하게 추가(BORDER_CONSTANT)
    sticker = cv2.copyMakeBorder(np.array(sticker), 100, 100, 100, 100, cv2.BORDER_CONSTANT,value=[255,255,255])
    sticker = np.array(Image.fromarray(sticker).resize(sticker_size))



    # landamarks 중 30번 째 Point
    nose_x = landmarks[30][0]
    nose_y = landmarks[30][1]

    # 스티커 이미지의 가세, 세로 길이 계산
    sticker_width = sticker_size[0]
    sticker_height = sticker_size[1]

    # nose좌표를 기준으로 스티커의 가로,세로의 절반 값 만큼 좌,상으로 이동
    sticker_x = nose_x - (sticker_width // 2)
    sticker_y = nose_y - (sticker_height // 2)


    example_image = image_show.copy()

    normal_image = image_show.copy()
    sticker_area = normal_image[sticker_y:sticker_y + sticker_height,
                                sticker_x:sticker_x + sticker_width]

    normal_image[sticker_y:sticker_y + sticker_height, 
                sticker_x:sticker_x + sticker_width] = np.where(sticker==255, sticker_area, sticker)



    x1, y1 = landmarks[0][0], landmarks[0][1] # Landmark 0번
    x2, y2 = landmarks[16][0], landmarks[16][1] # Landmark 16번
    dx, dy = (x2 - x1),  -(y2 - y1) # y대칭 적용
    theta = math.atan2(dy, dx) 

    matrix_rotation = cv2.getRotationMatrix2D((sticker_size[0]//2, sticker_size[1]//2), np.degrees(theta), 1)
    rotated_sticker = cv2.warpAffine(sticker, matrix_rotation, 
                                (sticker_size[0], sticker_size[1]), 
                                    borderValue = [255, 255, 255, 255])



    rotated_image = image_show.copy()
    sticker_area = rotated_image[sticker_y:sticker_y + sticker_height,
                        sticker_x:sticker_x + sticker_width]

    rotated_image[sticker_y:sticker_y + sticker_height, 
                sticker_x:sticker_x + sticker_width] = np.where(rotated_sticker==255, sticker_area, rotated_sticker)


    # Rotated된 스티커와 동일한 각도로 좌표 회전시키는 함수
    def rotate_point(point, radians, origin=(0, 0)):
        x, y = point        # 바꾸려는 좌표
        ox, oy = origin     # 회전 시 기준이 되었던 좌표(sticker 이미지의 중심) != 고양이 코

        qx = ox + math.cos(radians) * (x - ox) + math.sin(radians) * (y - oy)
        qy = oy + -math.sin(radians) * (x - ox) + math.cos(radians) * (y - oy)

        return int(qx), int(qy)

    # 고양이 코 중심 좌표 구하는 함수(위, 아래 좌표)
    def get_sticker_center(sticker_x_y, sticker_size, theta=theta):
        sticker_x, sticker_y = sticker_x_y     # 얼굴 이미지에서 sticker의 좌상단 좌표

        sticker_center_up_origin = (sticker_size[0] * 0.5, sticker_size[1] * 0.44)     # 스티커에서 고양이 코 위 좌표(비율)
        sticker_center_down_origin = (sticker_size[0] * 0.5, sticker_size[1] * 0.52)   # 스티커에서 고양이 코 아래 좌표(비율)

        # 고양이 코 중심 2개 좌표(위, 아래) Rotated 적용(위에서 정의한 rotate_point 함수 사용)
        sticker_c_up_rotated = rotate_point(sticker_center_up_origin, theta, origin=(sticker_size[0]//2, sticker_size[1]//2)) 
        sticker_c_down_rotated = rotate_point(sticker_center_down_origin, theta, origin=(sticker_size[0]//2, sticker_size[1]//2))

        # 스티커 이미지의 좌표를 얼굴 이미지의 좌표로 변환
        sticker_center_up = (sticker_x + int(sticker_c_up_rotated[0]), sticker_y + int(sticker_c_up_rotated[1]))
        sticker_center_down = (sticker_x + int(sticker_center_down_origin[0]), sticker_y + int(sticker_center_down_origin[1]))

        return sticker_center_up, sticker_center_down

    # 양이 스티커의 수엽 끝 부분 구하는 함수(오른쪽 3개)
    def get_whisker_point(sticker_x_y, sticker_size, theta=theta):
        sticker_x, sticker_y = sticker_x_y     # 얼굴 이미지에서 sticker의 좌상단 좌표

        whisker1_origin = (sticker_size[0] * 0.81, sticker_size[1] * 0.44)   # 스티커에서 고양이 수염 끝부분(위)
        whisker2_origin = (sticker_size[0] * 0.86, sticker_size[1] * 0.53)   # 스티커에서 고양이 수염 끝부분(중간)
        whisker3_origin = (sticker_size[0] * 0.74, sticker_size[1] * 0.58)   # 스티커에서 고양이 수염 끝부분(아래)

        # 고양이 수염 끝 부분 좌표(위,중심,아래) Rotated 적용(위에서 정의한 rotate_point 함수 사용)
        whisker1_rotated = rotate_point(whisker1_origin, theta, origin=(sticker_size[0]//2, sticker_size[1]//2)) # origin
        whisker2_rotated = rotate_point(whisker2_origin, theta, origin=(sticker_size[0]//2, sticker_size[1]//2)) # == 스티커 중심 좌표
        whisker3_rotated = rotate_point(whisker3_origin, theta, origin=(sticker_size[0]//2, sticker_size[1]//2))

        # 스티커 이미지의 좌표를 얼굴 이미지의 좌표로 변환
        whisker1 = (sticker_x + int(whisker1_rotated[0]), sticker_y + int(whisker1_rotated[1]))
        whisker2 = (sticker_x + int(whisker2_rotated[0]), sticker_y + int(whisker2_rotated[1]))
        whisker3 = (sticker_x + int(whisker3_rotated[0]), sticker_y + int(whisker3_rotated[1]))

        return whisker1, whisker2, whisker3


    # 수염 끝 부분 추출(오른쪽 3개)
    whisker1, whisker2, whisker3 = get_whisker_point((sticker_x, sticker_y), sticker_size)
    nose = (nose_x, nose_y) # landmark 30번 좌표(코)

    #d = np.linalg.norm(np.array(landmarks[15]) - np.array(whisker1))
    d = np.linalg.norm(np.array(landmarks[15]) - np.array(nose)) # 코와 턱 사이의 길이
    dt = 0.8 * d  # 코와 (새로운 좌표가 있으면 하는 좌표)까지의 길이

    # 각 수염별로 새로운 좌표 추출
    # 1번과 3번 수염의 경우 2번 수염보다 짧기 때문에 0.8 적용
    new_place1 = (dt * 0.8)  * ((np.array(landmarks[15]) - np.array(nose)) / d) + np.array(nose) 
    new_place2 = dt * ((np.array(landmarks[14]) - np.array(nose)) / d) + np.array(nose)
    new_place3 = (dt * 0.8) * ((np.array(landmarks[13]) - np.array(nose)) / d) + np.array(nose)

    # 시각화
    example_image = rotated_image.copy()



    new_sticker = rotated_sticker.copy() # Rotated된 스티커 이미지의 원본 파손 방지

    # 얼굴 이미지에서 기존 수염에서 늘어난 지점까지의 비율을 계산
    whisker_to_new_ratio1 = np.array(new_place1) / np.array(whisker1) # 위 수염
    whisker_to_new_ratio3 = np.array(new_place3) / np.array(whisker3) # 아래 수염

    # 스티커 내에서 변환되기 전 수염의 좌표 구하기
    whisker_in_sticker1, whisker_in_sticker2, whisker_in_sticker3 = get_whisker_point((0, 0), sticker_size)
    sticker_center_up, sticker_center_down = get_sticker_center((0, 0), sticker_size)

    ########################################################################################################
    # 원근감이 반대로 작용하는 문제 해결
    left = (np.array(landmarks[30][0]) - np.array(landmarks[2][0]))     # 왼쪽 턱과 코까지의 길이
    right = (np.array(landmarks[14][0]) - np.array(landmarks[30][0]))   # 오른쪽 턱과 코까지의 길이
    ratio = left/right     # 왼쪽과 오른쪽 얼굴의 길이 비율                                  
    if ratio < 0.7:
    weight = -0.04    # weight값(임의의 값)
    elif ratio > 1.3:
    weight = 0.04      # weight값(임의의 값)
    else:
    weight = 0         # weight값(임의의 값)

    # 변환되는 좌표(새로운 좌표)에 weight값 적용
    new_in_sticker1 = (whisker_in_sticker1 * (whisker_to_new_ratio3 + np.array([0, weight]))).astype(np.uint32)
    new_in_sticker3 = (whisker_in_sticker3 * (whisker_to_new_ratio1 + np.array([0, -weight]))).astype(np.uint32)


    # 원본 이미지(rotated_sticker)에서의 좌표(고양이 코 위,아래 + 오른쪽 1,3번 수염)
    src_p = np.float32([
    list(sticker_center_up),
    list(sticker_center_down),
    list(whisker_in_sticker3), 
    list(whisker_in_sticker1)])

    # 변환되는 좌표(rotated_sticker)에서의 좌표(고양이 코 위,아래 + 오른쪽 1,3번 수염)
    dst_p = np.float32([
    list(sticker_center_up),
    list(sticker_center_down),
    list(new_in_sticker3), 
    list(new_in_sticker1)])

    # Perspective transform Matrix 구하기 및 적용
    perspective_matrix = cv2.getPerspectiveTransform(src_p, dst_p)
    trans_sticker = cv2.warpPerspective(rotated_sticker, perspective_matrix, 
                                    (rotated_sticker.shape[1], rotated_sticker.shape[0]), 
                                    borderValue = [255, 255, 255, 255])


    result = image_show.copy()

    sticker_area = result[sticker_y:sticker_y + sticker_height,
                        sticker_x:sticker_x + sticker_width]
    result[sticker_y:sticker_y + sticker_height, 
        sticker_x:sticker_x + sticker_width] = np.where(trans_sticker==255, sticker_area, trans_sticker)
    
    return result