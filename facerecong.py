import cv2
import dlib
import numpy as np


face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def enlarge_eyes(image, landmarks, left_eye_points, right_eye_points, scale=2):
    # 获取眼睛区域
    left_eye = landmarks[left_eye_points[0]:left_eye_points[1]]
    right_eye = landmarks[right_eye_points[0]:right_eye_points[1]]

    # 计算眼睛的中心点
    left_eye_center = np.mean(left_eye, axis=0).astype(int)
    right_eye_center = np.mean(right_eye, axis=0).astype(int)

    # 计算眼睛的宽度和高度
    left_eye_width = int(np.linalg.norm(left_eye[0] - left_eye[3]) * scale)
    right_eye_width = int(np.linalg.norm(right_eye[0] - right_eye[3]) * scale)

    # 放大眼睛
    left_eye_resized = cv2.resize(image[left_eye_center[1]-left_eye_width//2:left_eye_center[1]+left_eye_width//2,
                                      left_eye_center[0]-left_eye_width//2:left_eye_center[0]+left_eye_width//2],
                                   (left_eye_width, left_eye_width))
    right_eye_resized = cv2.resize(image[right_eye_center[1]-right_eye_width//2:right_eye_center[1]+right_eye_width//2,
                                       right_eye_center[0]-right_eye_width//2:right_eye_center[0]+right_eye_width//2],
                                    (right_eye_width, right_eye_width))

    # 将放大后的眼睛放回原图
    image[left_eye_center[1]-left_eye_width//2:left_eye_center[1]+left_eye_width//2,
          left_eye_center[0]-left_eye_width//2:left_eye_center[0]+left_eye_width//2] = left_eye_resized
    image[right_eye_center[1]-right_eye_width//2:right_eye_center[1]+right_eye_width//2,
          right_eye_center[0]-right_eye_width//2:right_eye_center[0]+right_eye_width//2] = right_eye_resized

    return image


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    for face in faces:
        landmarks = landmark_predictor(gray, face)
        landmarks = np.array([(p.x, p.y) for p in landmarks.parts()])

        frame = enlarge_eyes(frame, landmarks, (36, 42), (42, 48), scale=2)

    cv2.imshow("Enlarged Eyes", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
