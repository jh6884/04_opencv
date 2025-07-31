# 모폴로지 연산

import cv2
import numpy as np

## 열림 / 닫힘 연산 (침식-팽창, 팽창-침식)
img1 = cv2.imread('../img/morph_dot.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('../img/morph_hole.png', cv2.IMREAD_GRAYSCALE)    

# 구조화 요소 커널, 사각형 (5x5) 생성
k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
# 열림 연산 적용
opening = cv2.morphologyEx(img1, cv2.MORPH_OPEN, k)
# 닫힘 연산 적용
closing = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, k)


## 그래디언트 연산(팽창 결과값에서 침식 결과값 빼기)

img = cv2.imread('../img/morphological.png')

# 구조화 요소 커널, 사각형 (3x3) 생성
k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# 열림 연산 적용
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, k)

# 결과 출력
merged = np.hstack((img, gradient))
merged1 = np.hstack((img1, opening))
merged2 = np.hstack((img2, closing))
merged3 = np.vstack((merged1, merged2))
cv2.imshow('gradient', merged)
cv2.imshow('opening, closing', merged3)
cv2.waitKey(0)
cv2.destroyAllWindows()
