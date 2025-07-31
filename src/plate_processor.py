# 번호판 이미지 전처리

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# 번호판 파일 불러오기
def load_extracted_plate(plate_name):
    plate_path = f"../extracted_plates/{plate_name}.png"
    if os.path.exists(plate_path):
        plate_img = cv2.imread(plate_path)
        print(f'번호판 이미지 로드 성공 {plate_img.shape}')
        return plate_img
    else:
        print("파일을 찾을 수 없습니다.")
        return None
    
# 번호판 파일 그레이스케일로 변경
def convert_to_grayScale(target_img):
    gray_plate = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    # 결과 비교 시각화
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Extracted Plate')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(gray_plate, cmap='gray')
    plt.title('Grayscale Plate')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    return gray_plate

# 대비 최대화
def contrast_maximize(gray_plate):
    # 모폴로지 연산용 구조화 요소
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    # 모폴로지 연산(tophat, blackhat)
    tophat = cv2.morphologyEx(gray_plate, cv2.MORPH_TOPHAT, k)
    blackhat = cv2.morphologyEx(gray_plate, cv2.MORPH_BLACKHAT, k)
    # 대비 향상 적용
    # enhanced = cv2.add(gray_plate, tophat)
    # enhanced = cv2.subtract(gray_plate, blackhat)
    # enhanced = cv2.equalizeHist(enhanced)
    enhanced = cv2.equalizeHist(gray_plate)
    # 결과 비교 시각화
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(gray_plate, cmap='gray')
    plt.title('Original Gray')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(tophat, cmap='gray')
    plt.title('Top Hat')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(blackhat, cmap='gray')
    plt.title('Black Hat')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(enhanced, cmap='gray')
    plt.title('Enhanced Contrast')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return enhanced

# 임계처리
def threshold_plate(enhanced_plate):
    blurred = cv2.GaussianBlur(enhanced_plate, (3,3), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh

# 윤곽선 처리
def find_contour(thresh):
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 결과 시각화용 이미지 생성
    height, width = thresh.shape
    contour_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    # 모든 윤곽선을 다른 색으로 그리기
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
    for i, contour in enumerate(contours):
        color = colors[i % len(colors)]
        cv2.drawContours(contour_img, [contour], -1, color, 2)
        # 윤곽선 번호 표시
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cX = int(M['m10']/M['m00'])
            cY = int(M['m01']/M['m00'])
            cv2.putText(contour_img, str(i+1), (cX-5, cY+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    # 결과 시각화
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(thresh, cmap='gray')
    plt.title('Binary Plate')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(contour_img)
    plt.title(f'Contours Detected: {len(contours)}')
    plt.axis('off')
# 윤곽선 정보 표시

    plt.subplot(1, 3, 3)

    contour_info = np.zeros((height, width, 3), dtype=np.uint8)

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        # 경계 사각형 그리기
        cv2.rectangle(contour_info, (x, y), (x+w, y+h), colors[i % len(colors)], 1)

        # 면적 정보 표시 (작은 글씨로)
        cv2.putText(contour_info, f'A:{int(area)}', (x, y-2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
    
    plt.imshow(contour_info)
    plt.title('Bounding Rectangles')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    # 윤곽선 정보 출력
    print("=== 윤곽선 검출 결과 ===")
    print(f"총 윤곽선 개수: {len(contours)}")

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        print(f"윤곽선 {i+1}: 면적={area:.0f}, 크기=({w}×{h}), 비율={aspect_ratio:.2f}")

    return contours, contour_img

def prepare_for_next_step(contours):
    print("=== 다음 단계 준비 ===")
    # 윤곽선이 충분히 검출되었는지 확인
    if len(contours) < 5:
        print("윤곽선이 적게 검출되었습니다. 전처리 단계를 재검토하세요.")
    elif len(contours) > 20:
        print("윤곽선이 너무 많이 검출되었습니다. 노이즈 제거가 필요할 수 있습니다.")
    else:
        print("적절한 수의 윤곽선이 검출되었습니다.")

    # 잠재적 글자 후보 개수 추정
    potential_chars = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if 30 < area < 2000:  # 글자 크기 범위 추정
            potential_chars += 1
    print(f"잠재적 글자 후보: {potential_chars}개")

    return potential_chars

def save_processed_results(plate_name, gray_plate, enhanced_plate, thresh_plate, contour_result):
    # 저장 폴더 생성
    save_dir = "../processed_plates"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 단계별 결과 저장
    cv2.imwrite(f'{save_dir}/{plate_name}_1_gray.png', gray_plate)
    cv2.imwrite(f'{save_dir}/{plate_name}_2_enhanced.png', enhanced_plate)  
    cv2.imwrite(f'{save_dir}/{plate_name}_3_threshold.png', thresh_plate)
    cv2.imwrite(f'{save_dir}/{plate_name}_4_contours.png', contour_result)

    print("결과 저장 완료")

# 전체 처리 파이프라인
def process_extracted_plates(plate_name):
    print(f"{plate_name} 처리 시작")
    # 이미지 로드
    plate_img = load_extracted_plate(plate_name)
    if plate_img is None:
        return None
    
    # 그레이스케일 변환
    gray_plate = convert_to_grayScale(plate_img)

    # 대비 최대화
    enhanced_plate = contrast_maximize(gray_plate)

    # 임계처리
    thresh_plate = threshold_plate(enhanced_plate)

    # 윤곽선 검출
    contours, contour_result = find_contour(thresh_plate)

    # 결과 저장
    save_processed_results(plate_name, gray_plate, enhanced_plate, thresh_plate, contour_result)

    # 처리 결과 요약
    potential_chars = prepare_for_next_step(contours)
    print(f"처리 완료, 검출된 윤곽선 {len(contours)}개, 잠재적 글자 {potential_chars}개")
    return {
        'original': plate_img,
        'gray': gray_plate, 
        'enhanced': enhanced_plate,
        'threshold': thresh_plate,
        'contours': len(contours),
        'potential_chars': potential_chars,
        'contour_result': contour_result
    }

# 배치 처리
def batch_process_plates():
    plate_dir = '../extracted_plates'

    if not os.path.exists(plate_dir):
        print(f"폴더를 찾을 수 없습니다: {plate_dir}")
        return {}

    plate_files = [f for f in os.listdir(plate_dir) if f.endswith('.png')]

    if len(plate_files) == 0:
        print("처리할 번호판 이미지가 없습니다.")
        return {}
    
    results = {}

    for plate_file in plate_files:
        plate_name = plate_file.replace('.png', '')
        result = process_extracted_plates(plate_name)
        if result:
            results[plate_name] = result
        
    print(f"\n=== 전체 처리 완료: {len(results)}개 번호판 ===")

    return results

batch_process_plates()