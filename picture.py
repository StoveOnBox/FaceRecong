import cv2
import numpy as np

def detect_faces(image, face_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces


def enlarge_eyes(image, faces, scale_factor):
    for (x, y, w, h) in faces:
        roi_color = image[y:y + h, x:x + w]
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(roi_color)

        for (ex, ey, ew, eh) in eyes:
            center = (ex + ew // 2, ey + eh // 2)
            radius_x, radius_y = int(scale_factor * ew / 2), int(scale_factor * eh / 2)
            new_center = (x + center[0], y + center[1])
            axes = (radius_x, radius_y)
            angle = 0
            startAngle = 0
            endAngle = 360
            color = (255, 0, 0)
            thickness = 2

            cv2.ellipse(image, new_center, axes, angle, startAngle, endAngle, color, thickness)

    return image


def main(input_image_path, output_image_path, scale_factor):
    image = cv2.imread(input_image_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = detect_faces(image, face_cascade)

    if len(faces) > 0:
        result = enlarge_eyes(image, faces, scale_factor)
        cv2.imwrite(output_image_path, result)
        print("成功处理图片，结果已保存到", output_image_path)
    else:
        print("未检测到人脸，请尝试其他图片。")


if __name__ == '__main__':
    input_image_path = 'input.jpg'  # 输入图片路径
    output_image_path = 'output.jpg'  # 输出图片路径
    scale_factor = 1.5  # 眼部放大倍数

    main("C:/UserDocument/codes/PY/FaceRecong/input_image.jpg", "C:/UserDocument/codes/PY/FaceRecong/output_image.jpg",
         4)

