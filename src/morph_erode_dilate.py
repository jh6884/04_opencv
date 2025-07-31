# 침식 팽창 연산

import cv2
import numpy as np

img1 = cv2.imread('../img/morph_dot.png')
img2 = cv2.imread('../img/morph_hole.png')

# 구조화 요소 커널, 사각형 (3x3) 생성
k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# 침식 연산 적용
erosion = cv2.erode(img1, k)
# 팽창 연산 적용
dst = cv2.dilate(img2, k)

#결과 출력
merged1 = np.hstack((img1, erosion))
merged2 = np.hstack((img2, dst))
cv2.imshow('Erosion', merged1)
cv2.imshow('Dilation', merged2)
cv2.waitKey(0)
cv2.destroyAllWindows()