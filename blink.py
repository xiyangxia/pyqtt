#coding=utf-8  
import numpy as np 
import cv2
import dlib
from scipy.spatial import distance as dist
import os
from imutils import face_utils

current_path = os.getcwd()  # 获取当前路径
predictor_path = current_path + "\\model\\shape_predictor_68_face_landmarks.dat"  # shape_predictor_68_face_landmarks.dat是进行人脸标定的模型，它是基于HOG特征的，这里是他所在的路径
face_directory_path = current_path + "\\faces\\"  # 存放人脸图片的路径
detector = dlib.get_frontal_face_detector()  # 获取人脸分类器
predictor = dlib.shape_predictor(predictor_path)  # 获取人脸检测器


EYE_AR_THRESH = 0.3# EAR阈值
EYE_AR_CONSEC_FRAMES = 3# 当EAR小于阈值时，接连多少帧一定发生眨眼动作

# 对应特征点的序号
RIGHT_EYE_START = 37 - 1
RIGHT_EYE_END = 42 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1

def eye_aspect_ratio(self,eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

frame_counter = 0# 连续帧计数 
blink_counter = 0# 眨眼计数
cap = cv2.VideoCapture(1)
while(1):
    ret, img = cap.read()# 读取视频流的一帧
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    b,g,r = cv2.split(img)  # 分离三个颜色通道
    img2 = cv2.merge([r, g, b])  # 融合三个颜色通道生成新图片
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# 转成灰度图像
    rects = detector(img2, 0)# 人脸检测
    for rect in rects:# 遍历每一个人脸
        print('-'*20)
        shape = predictor(img2, rect)# 检测特征点
        points = face_utils.shape_to_np(shape)# convert the facial landmark (x, y)-coordinates to a NumPy array
        leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]# 取出左眼对应的特征点
        rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]# 取出右眼对应的特征点
        leftEAR = eye_aspect_ratio(leftEye)# 计算左眼EAR
        rightEAR = eye_aspect_ratio(rightEye)# 计算右眼EAR
        print('leftEAR = {0}'.format(leftEAR))
        print('rightEAR = {0}'.format(rightEAR))

        ear = (leftEAR + rightEAR) / 2.0# 求左右眼EAR的均值

        leftEyeHull = cv2.convexHull(leftEye)# 寻找左眼轮廓
        rightEyeHull = cv2.convexHull(rightEye)# 寻找右眼轮廓
        cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)# 绘制左眼轮廓
        cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)# 绘制右眼轮廓

        # 如果EAR小于阈值，开始计算连续帧，只有连续帧计数超过EYE_AR_CONSEC_FRAMES时，才会计做一次眨眼
        if ear < EYE_AR_THRESH:
            frame_counter += 1
        else:
            if frame_counter >= EYE_AR_CONSEC_FRAMES:
                blink_counter += 1
            frame_counter = 0

        # 在图像上显示出眨眼次数blink_counter和EAR
        cv2.putText(img, "Blinks:{0}".format(blink_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(img, "EAR:{:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)


    cv2.imshow("Frame", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
