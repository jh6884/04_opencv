# 마우스와 원근 변환으로 문서 스캔 효과 내기

import cv2
import numpy as np
import os

img_name = 'car_02' # 이미지명

win_name = "License Plate Extractor"
img = cv2.imread(f'../img/{img_name}.jpg')
rows, cols = img.shape[:2]
draw = img.copy() # 점을 그릴 이미지 사본
pts_cnt = 0 # 수집한 좌표의 개수
pts = np.zeros((4,2), dtype=np.float32)

def onMouse(event, x, y, flags, param): # 마우스 이벤트 콜백 함수
    global pts_cnt
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(draw, (x,y), 5, (0, 255, 0), -1) # 클릭한 곳에 동그라미 표시
        cv2.imshow(win_name, draw)

        pts[pts_cnt] = [x,y] # 좌표 배열에 저장
        pts_cnt += 1 # 수집한 좌표 개수에 +1

        if pts_cnt == 4: # 좌표 4개 수집 시
            cv2.destroyWindow(win_name)
            sm = pts.sum(axis=1) # 좌표 각각 x+y 계산
            diff = np.diff(pts, axis = 1) # 좌표 각각 x-y 계산

            topLeft = pts[np.argmin(sm)]
            bottomRight = pts[np.argmax(sm)]
            topRight = pts[np.argmin(diff)]
            bottomLeft = pts[np.argmax(diff)]

            # 변환 전 4개 좌표
            pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

            # 변환 후 이미지의 크기 고정
            width = 300
            height = 150

            # 변환 후 4개 좌표
            pts2 = np.float32([[0,0], [width-1, 0], [width-1, height-1], [0, height-1]])

            # 변환 행렬 계산
            mtrx = cv2.getPerspectiveTransform(pts1, pts2)

            # 원근 변환 적용
            result = cv2.warpPerspective(img, mtrx, (int(width), int(height)))

            # 변환된 이미지 저장
            existing_files = len(os.listdir('../extracted_plates'))
            file_name = f'../extracted_plates/plate_{existing_files+1:02d}.jpg'
            save = cv2.imwrite(file_name, result)
            if save:
                print('File saved! Press any key to exit program')
                cv2.imshow('Extracted Plate', result)
            else:
                print('Save failed')


cv2.imshow(win_name, img)
cv2.setMouseCallback(win_name, onMouse)
cv2.waitKey(0)
cv2.destroyAllWindows()