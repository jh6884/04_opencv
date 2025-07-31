# 이미지 회전

import cv2
import numpy as np

img = cv2.imread('../img/fish.jpg')
rows, cols = img.shape[0:2]

# 라디안 각도 계산
d45 = 45.0 * np.pi / 180 # 45도
d90 = 90.0 * np.pi / 180 # 90도

# 회전을 위한 변환 행렬 생성
m45 = np.float32( [[ np.cos(d45), -1* np.sin(d45), rows//2],
                    [np.sin(d45), np.cos(d45), -1*cols//4]])
m90 = np.float32( [[ np.cos(d90), -1* np.sin(d90), rows],
                    [np.sin(d90), np.cos(d90), 0]])

# 회전 변환 행렬 적용
r45 = cv2.warpAffine(img,m45,(cols,rows))
r90 = cv2.warpAffine(img,m90,(rows,cols))

## 함수를 사용해 변환 행렬 구하기

# 회전축:중앙, 각도:45, 배율:0.5
m45_2 = cv2.getRotationMatrix2D((cols/2,rows/2),45,0.5) 
# 회전축:중앙, 각도:90, 배율:1.5
m90_2 = cv2.getRotationMatrix2D((cols/2,rows/2),90,1.5)

# 회전 변환 행렬 적용
img45 = cv2.warpAffine(img, m45_2,(cols, rows))
img90 = cv2.warpAffine(img, m90_2,(cols, rows))

# 결과 출력
cv2.imshow("origin", img)
cv2.imshow("45", r45)
cv2.imshow("90", r90)
cv2.imshow("45_2", img45)
cv2.imshow("90_2", img90)
cv2.waitKey(0)
cv2.destroyAllWindows()