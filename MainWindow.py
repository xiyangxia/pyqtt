import pymysql
import json
from face_train import Model
from UI import Ui_MainWindow
#from UI_new import Ui_MainWindow
from Log import Ui_Dialog
import sys
from MyTipWindow import Message
import face_recognition
import threading
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QSystemTrayIcon, QMessageBox, QDialog
from PyQt5.QtCore import QTimer, QCoreApplication
from PyQt5.QtGui import QPixmap
import cv2
import qimage2ndarray
import time
import dlib
import os
import pandas as pd
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
from keras.models import load_model
import datetime
import ChineseText

#  opencv 检测人脸
face_detector = cv2.CascadeClassifier('D:/OPENCV/sources/data/haarcascades/haarcascade_frontalface_default.xml')
gender_classifier = load_model("model/simple_CNN-gender.hdf5")
emotion_classifier = load_model("model/simple_CNN-emotion.hdf5")
gender_labels = {0: 'girl', 1: 'boy'}
emotion_labels = {
    0: 'angry',
    1: 'hate',
    2: 'terror',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'calm'
}


ids = []
face_names = []
face_codings = []

face_sampes = []
person_list = os.listdir("faces/")
for i in range(len(person_list)):
    face_coding_mean = []
    person_name = os.listdir("faces/" + "person_" + str(i + 1))
    #print(person_name)
    # print(person_name[len(person_name)-1])

    for j in range(len(person_name)):

        img_path = "faces/" + "person_" + str(i + 1) + "/" + person_name[j]
        #print(img_path)
        face_img = face_recognition.load_image_file(img_path)
        # opencv人脸识别

        PIL_img = Image.open(img_path).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        faces = face_detector.detectMultiScale(img_numpy)
        # print(len(faces))
        for x, y, w, h in faces:
            face_sampes.append(img_numpy[y:y + h, x:x + w])
            ids.append(i + 1)
        face_coding_mean.append(face_recognition.face_encodings(face_img)[0])
    face_codings.append(np.array(face_coding_mean).mean(axis=0))
    # face_names.append(person_name[0][:person_name[0].index(".")])
    face_names.append(person_name[len(person_name) - 1][:person_name[len(person_name) - 1].index(".")])
font = cv2.FONT_HERSHEY_DUPLEX
current_path = os.getcwd()  # 获取当前路径
predictor_path = current_path + "\\model\\shape_predictor_68_face_landmarks.dat"  # shape_predictor_68_face_landmarks.dat是进行人脸标定的模型，它是基于HOG特征的，这里是他所在的路径
face_directory_path = current_path + "\\faces\\"  # 存放人脸图片的路径
detector = dlib.get_frontal_face_detector()  # 获取人脸分类器
predictor = dlib.shape_predictor(predictor_path)  # 获取人脸检测器
facerec = dlib.face_recognition_model_v1("model/dlib_face_recognition_resnet_model_v1.dat")


path_features_known_csv = "features_all.csv"
csv_rd = pd.read_csv(path_features_known_csv, header=None)
# 用来存放所有录入人脸特征的数组
# The array to save the features of faces in the database
features_known_arr = []
# 2. 读取已知人脸数据
# Print known faces
for i in range(csv_rd.shape[0]):
    features_someone_arr = []
    for j in range(0, len(csv_rd.iloc[i])):
        features_someone_arr.append(csv_rd.iloc[i][j])
    features_known_arr.append(features_someone_arr)
# #print("Faces in Database：", len(features_known_arr))
# str_data = 'Faces in Database: '+str(len(features_known_arr))
# print(str_data)
# self.MsgTE.setPlainText(str_data)

# 每次启动都要执行，浪费效率
# dlib 获取特征
#os.system("python36 getfacefeatures_to_csv.py")#程序启动后先提取所有人脸特征，防止后台手动添加人脸

# opencv训练
# print(face_sampes)
#print(ids)
opencv_recognizer = cv2.face.LBPHFaceRecognizer_create()
# opencv_recognizer.train(face_sampes, np.array(ids))
# opencv_recognizer.write('train/train.yml')
name = ""
EYE_AR_THRESH = 0.18  # EAR阈值
EYE_AR_CONSEC_FRAMES = 4  # 当EAR小于阈值时，接连多少帧一定发生眨眼动作

# 对应特征点的序号
RIGHT_EYE_START = 37 - 1
RIGHT_EYE_END = 42 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1


class childWindow(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.child=Ui_Dialog()

    def show(self):
        #self.child.loaddate()
        self.child.setupUi(self)
        super().show()



class MainWindow(QMainWindow, Ui_MainWindow):
    camera = cv2.VideoCapture(0)

    id = 0
    flag_1 = True
    flag_2 = True
    flag_3 = True

    #cnn
    model = Model()
    with open('contrast_table', 'r') as f:
        contrast_table = json.loads(f.read())
    model.load_model(file_path='./model/face.model')
    def __del__(self):
        try:
            self.camera.release()  # 释放资源
        except:
            return

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.PrepSliders()
        self.PrepWidgets()
        self.PrepParameters()
        self.CallBackFunctions()
        self.Timer = QTimer()
        self.Timer.timeout.connect(self.TimerOutFun)

    def PrepSliders(self):
        self.RedColorSld.valueChanged.connect(self.RedColorSpB.setValue)
        self.RedColorSpB.valueChanged.connect(self.RedColorSld.setValue)
        self.GreenColorSld.valueChanged.connect(self.GreenColorSpB.setValue)
        self.GreenColorSpB.valueChanged.connect(self.GreenColorSld.setValue)
        self.BlueColorSld.valueChanged.connect(self.BlueColorSpB.setValue)
        self.BlueColorSpB.valueChanged.connect(self.BlueColorSld.setValue)
        self.ExpTimeSld.valueChanged.connect(self.ExpTimeSpB.setValue)
        self.ExpTimeSpB.valueChanged.connect(self.ExpTimeSld.setValue)
        self.GainSld.valueChanged.connect(self.GainSpB.setValue)
        self.GainSpB.valueChanged.connect(self.GainSld.setValue)
        self.BrightSld.valueChanged.connect(self.BrightSpB.setValue)
        self.BrightSpB.valueChanged.connect(self.BrightSld.setValue)
        self.ContrastSld.valueChanged.connect(self.ContrastSpB.setValue)
        self.ContrastSpB.valueChanged.connect(self.ContrastSld.setValue)

    def PrepWidgets(self):
        self.PrepCamera()
        self.StopBt.setEnabled(False)
        self.RecordBt.setEnabled(False)
        self.GrayImgCkB.setEnabled(False)
        self.RedColorSld.setEnabled(False)
        self.RedColorSpB.setEnabled(False)
        self.GreenColorSld.setEnabled(False)
        self.GreenColorSpB.setEnabled(False)
        self.BlueColorSld.setEnabled(False)
        self.BlueColorSpB.setEnabled(False)
        self.ExpTimeSld.setEnabled(False)
        self.ExpTimeSpB.setEnabled(False)
        self.GainSld.setEnabled(False)
        self.GainSpB.setEnabled(False)
        self.BrightSld.setEnabled(False)
        self.BrightSpB.setEnabled(False)
        self.ContrastSld.setEnabled(False)
        self.ContrastSpB.setEnabled(False)
        # self.pushButton.setEnabled(True)

    def PrepCamera(self):
        try:
            # self.camera=cv2.VideoCapture(0)
            self.MsgTE.clear()
            self.MsgTE.append('Oboard camera connected.')
            self.MsgTE.setPlainText()
        except Exception as e:
            self.MsgTE.clear()
            self.MsgTE.append(str(e))

    def PrepParameters(self):
        self.RecordFlag = 0
        self.RecordPath = 'E:/pyqtt/'
        self.FilePathLE.setText(self.RecordPath)
        self.Image_num = 0
        self.R = 1
        self.G = 1
        self.B = 1

        self.ExpTimeSld.setValue(self.camera.get(15))
        self.SetExposure()
        self.GainSld.setValue(self.camera.get(14))
        self.SetGain()
        self.BrightSld.setValue(self.camera.get(10))
        self.SetBrightness()
        self.ContrastSld.setValue(self.camera.get(11))
        self.SetContrast()
        self.MsgTE.clear()

    def CallBackFunctions(self):
        self.FilePathBt.clicked.connect(self.SetFilePath)
        self.ShowBt.clicked.connect(self.StartCamera)
        self.StopBt.clicked.connect(self.StopCamera)
        self.RecordBt.clicked.connect(self.RecordCamera)
        self.ExitBt.clicked.connect(self.ExitApp)
        self.GrayImgCkB.stateChanged.connect(self.SetGray)
        self.ExpTimeSld.valueChanged.connect(self.SetExposure)
        self.GainSld.valueChanged.connect(self.SetGain)
        self.BrightSld.valueChanged.connect(self.SetBrightness)
        self.ContrastSld.valueChanged.connect(self.SetContrast)
        self.RedColorSld.valueChanged.connect(self.SetR)
        self.GreenColorSld.valueChanged.connect(self.SetG)
        self.BlueColorSld.valueChanged.connect(self.SetB)
        #self.showLog.clicked.connect(self.logView)


    def SetR(self):
        R = self.RedColorSld.value()
        self.R = R / 255

    def SetG(self):
        G = self.GreenColorSld.value()
        self.G = G / 255

    def SetB(self):
        B = self.BlueColorSld.value()
        self.B = B / 255

    def SetContrast(self):
        contrast_toset = self.ContrastSld.value()
        try:
            self.camera.set(11, contrast_toset)
            self.MsgTE.setPlainText('The contrast is set to ' + str(self.camera.get(11)))
        except Exception as e:
            self.MsgTE.setPlainText(str(e))

    def SetBrightness(self):
        brightness_toset = self.BrightSld.value()
        try:
            self.camera.set(10, brightness_toset)
            self.MsgTE.setPlainText('The brightness is set to ' + str(self.camera.get(10)))
        except Exception as e:
            self.MsgTE.setPlainText(str(e))

    def SetGain(self):
        gain_toset = self.GainSld.value()
        print(gain_toset)
        try:
            #14为增益，改为12饱和度
            self.camera.set(14, gain_toset)
            #print(self.camera.get(5))
            self.MsgTE.setPlainText('The gain is set to ' + str(self.camera.get(14)))
        except Exception as e:
            self.MsgTE.setPlainText(str(e))

    def SetExposure(self):
        exposure_time_toset = self.ExpTimeSld.value()
        print(exposure_time_toset)
        try:
            #15为曝光，改为13色调图像
            self.camera.set(15, exposure_time_toset)
            #print(self.camera.get(5))
            self.MsgTE.setPlainText('The exposure time is set to ' + str(self.camera.get(15)))
        except Exception as e:
            self.MsgTE.setPlainText(str(e))

    def SetGray(self):
        if self.GrayImgCkB.isChecked():
            self.RedColorSld.setEnabled(False)
            self.RedColorSpB.setEnabled(False)
            self.GreenColorSld.setEnabled(False)
            self.GreenColorSpB.setEnabled(False)
            self.BlueColorSld.setEnabled(False)
            self.BlueColorSpB.setEnabled(False)
        else:
            self.RedColorSld.setEnabled(True)
            self.RedColorSpB.setEnabled(True)
            self.GreenColorSld.setEnabled(True)
            self.GreenColorSpB.setEnabled(True)
            self.BlueColorSld.setEnabled(True)
            self.BlueColorSpB.setEnabled(True)

    def StartCamera(self):
        # ret,fram = self.camera.read()
        # cv2.imshow('video', fram)
        tag = self.ShowBt.text()
        if tag == '开始':
            self.ShowBt.setEnabled(False)
            #self.ShowBt.setText("查看日志")
            self.StopBt.setEnabled(True)
            self.RecordBt.setEnabled(True)
            self.GrayImgCkB.setEnabled(True)
            if self.GrayImgCkB.isChecked() == 0:
                self.RedColorSld.setEnabled(True)
                self.RedColorSpB.setEnabled(True)
                self.GreenColorSld.setEnabled(True)
                self.GreenColorSpB.setEnabled(True)
                self.BlueColorSld.setEnabled(True)
                self.BlueColorSpB.setEnabled(True)
            self.ExpTimeSld.setEnabled(True)
            self.ExpTimeSpB.setEnabled(True)
            self.GainSld.setEnabled(True)
            self.GainSpB.setEnabled(True)
            self.BrightSld.setEnabled(True)
            self.BrightSpB.setEnabled(True)
            self.ContrastSld.setEnabled(True)
            self.ContrastSpB.setEnabled(True)
            self.RecordBt.setText('录像')
            #原来是10，更改为50
            self.Timer.start(10)
            self.timelb = time.clock()
        elif tag == '保存到已有':
            face_path = "E:/pyqtt/faces/"
            file_path = QFileDialog.getSaveFileName(self, "保存文件", face_path,
                                                    "jpg files (*.jpg);;all files(*.*)")
            #print(file_path[0])
            cv2.imwrite(file_path[0], self.Image)
            self.MsgTE.clear()
            self.MsgTE.setPlainText('Image saved.')
        # elif tag == '查看日志':
        #     #self.camera.release()
        #     pixmap = QPixmap("C:\\Users\\Administrator\\Desktop\\biye\\beijing.jpg")  # 按指定路径找到图片，注意路径必须用双引号包围，不能用单引号
        #     self.DispLb.setPixmap(pixmap)  # 在label上显示图片
        #     self.DispLb.setScaledContents(True)  # 让图片自适应label大小
        #     self.ShowBt.setText("开始")
            #self.__init__()
            # log = childWindow()
            # log.show()


    def SetFilePath(self):
        dirname = QFileDialog.getExistingDirectory(self, "浏览", '.')
        if dirname:
            self.FilePathLE.setText(dirname)
            self.RecordPath = dirname + '/'

    def TimerOutFun(self):
        global name
        success, img = self.camera.read()
        if success:
            if self.checkBox.isChecked():
                name = "unknown"
                self.face_recognise(img)
                self.flag_2 = True
                self.flag_3 = True
                # t1 = threading.Thread(target=self.face_recognise, args=[img])
                # t1.start()
                #
                #
                # time.sleep(1)
                try:
                    if self.flag_1:
                        self.MsgTE.setPlainText('Based on face_recognition'+'\n'+'The man in camera maybe ' + str(name))
                    else:
                        self.MsgTE.setPlainText('The door is open,please come in!'+
                                                '\n'+'Based on face_recognition' +
                                                '\n' + 'The man in camera maybe '
                                                + str(name))
                except Exception as e:
                    self.MsgTE.setPlainText(str(e))
                # t1.stop()

            if self.checkBox_2.isChecked():
                self.landmark(img)
                # t2 = threading.Thread(target=self.landmark, args=[img])
                # t2.start()
                # time.sleep(1)

            if self.checkBox_3.isChecked():
                name = "unknown"
                self.dlib_recognise(img)
                self.flag_2 = True
                self.flag_1 = True
                # t3 = threading.Thread(target=self.dlib_recognise, args=[img])
                # t3.start()
                # time.sleep(1)
                try:
                    if self.flag_3:
                        self.MsgTE.setPlainText(
                            'Based on dlib' + '\n' + 'The man in camera maybe ' + str(name))
                    else:
                        self.MsgTE.setPlainText('The door is open,please come in!' +
                                                '\n' + 'Based on dlib' +
                                                '\n' + 'The man in camera maybe '
                                                + str(name))
                except Exception as e:
                    self.MsgTE.setPlainText(str(e))

            if self.checkBox_4.isChecked():
                self.blink_recognise(img)
                # msgBox = Message()
                # msgBox.setText("Hello!")
                # # msgBox.setIcon(QMessageBox::Information)
                # # msgBox.setStandardButtons(QMessageBox::Ok)
                # msgBox.autoClose = True
                # msgBox.timeout = 3
                # msgBox.show()
                # t4 = threading.Thread(target=self.blink_recognise, args=[img])
                # t4.start()
                # time.sleep(1)
            if self.checkBox_7.isChecked():
                name = "unknown"
                self.cnn_recognise(img)


            if self.checkBox_6.isChecked():
                name = "unknown"
                self.opencv_recognise(img)
                self.flag_1 = True
                self.flag_3 = True
                # t4 = threading.Thread(target=self.opencv_recognise, args=[img])
                # t4.start()
                # time.sleep(1)
                try:
                    if self.flag_2:
                        self.MsgTE.setPlainText(
                            'Based on opencv' + '\n' + 'The man in camera maybe ' + str(name))
                    else:
                        self.MsgTE.setPlainText('The door is open,please come in!' +
                                                '\n' + 'Based on opencv' +
                                                '\n' + 'The man in camera maybe '
                                                + str(name))
                except Exception as e:
                    self.MsgTE.setPlainText(str(e))

            self.Image = self.ColorAdjust(img)
            self.DispImg()
            self.Image_num += 1
            if self.RecordFlag:
                self.video_writer.write(img)
            ###计算帧率？
            if self.Image_num % 10 == 9:
                frame_rate = 10 / (time.clock() - self.timelb)
                self.FmRateLCD.display(frame_rate)
                self.timelb = time.clock()
                # size=img.shape
                self.ImgWidthLCD.display(self.camera.get(3))
                self.ImgHeightLCD.display(self.camera.get(4))

        else:
            self.MsgTE.clear()
            self.MsgTE.setPlainText('Image obtaining failed.')

    def cnn_recognise(self,img):
        frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faceRects = face_detector.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect
                # 截取脸部图像提交给模型识别这是谁
                image = img[y - 10: y + h + 10, x - 10: x + w + 10]
                probability, name_number = self.model.face_predict(image)
                #print(name_number)
                cname = self.contrast_table[str(name_number)]
                #print(cname)
                # print('name_number:', name_number)
                cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), thickness=2)

                # 文字提示是谁
                cv2.putText(img, cname, (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                # if probability > 0.7:
                #     cv2.putText(frame, name, (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                # else:
                #     cv2.putText(frame, 'unknow', (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # coding=utf-8
    # 中文乱码处理,未成功
    def cv2ImgAddText(self,img, text, left, top, textColor=(0, 255, 0), textSize=20):
        if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        fontText = ImageFont.truetype(
            "font/simsun.ttc", textSize, encoding="utf-8")
        draw.text((left, top), text, textColor, font=fontText)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    # 子窗口调用，未使用此方法
    def logView(self):
        # log = childWindow()
        # log.show()
        pass

    def opencv_recognise(self, img):
        global name
        imgCompose = cv2.imread("compose/maozi-1.jpg")
        # 读取训练文件
        opencv_recognizer.read('train/train.yml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        new_faces = face_detector.detectMultiScale(gray)

        for x, y, w, h in new_faces:
            face = img[(y - 60):(y + h + 60), (x - 30):(x + w + 30)]
            face = cv2.resize(face, (48, 48))
            face = np.expand_dims(face, 0)
            face = face / 255.0
            gender_label_arg = np.argmax(gender_classifier.predict(face))
            gender = gender_labels[gender_label_arg]

            gray_face = gray[(y):(y + h), (x):(x + w)]
            gray_face = cv2.resize(gray_face, (48, 48))
            gray_face = gray_face / 255.0
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            #gray_face = np.expand_dims(gray_face, -1)
            emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
            emotion = emotion_labels[emotion_label_arg]
            # print(emotion)
            #img = ChineseText.cv2ImgAddText(img, gender, x + h, y, (255, 255, 255), 30)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            id, confidence = opencv_recognizer.predict(gray[y:y + h, x:x + w])
            cv2.putText(img, face_names[id - 1], (x + 6, y + h - 6), font, 1.0, (0, 255, 255), 1)
            cv2.putText(img, gender, (x + h, y ), font, 1.0, (0, 255, 255), 1)
            cv2.putText(img, emotion, (x, y), font, 1.0, (0, 255, 255), 1)

            name = face_names[id - 1] + "\nThe confidence:" +str(int(confidence))

            ##合成帽子
            sp = imgCompose.shape
            imgComposeSizeH = int(sp[0] / sp[1] * w)
            if imgComposeSizeH > (y - 20):
                imgComposeSizeH = (y - 20)
            imgComposeSize = cv2.resize(imgCompose, (w, imgComposeSizeH), interpolation=cv2.INTER_NEAREST)
            top = (y - imgComposeSizeH - 20)
            if top <= 0:
                top = 0
            rows, cols, channels = imgComposeSize.shape
            roi = img[top:top + rows, x:x + cols]

            # Now create a mask of logo and create its inverse mask also
            img2gray = cv2.cvtColor(imgComposeSize, cv2.COLOR_RGB2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            # Now black-out the area of logo in ROI
            img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

            # Take only region of logo from logo image.
            img2_fg = cv2.bitwise_and(imgComposeSize, imgComposeSize, mask=mask)

            # Put logo in ROI and modify the main image
            dst = cv2.add(img1_bg, img2_fg)
            img[top:top + rows, x:x + cols] = dst

            #print(gender)
            #name = face_names[id - 1]
            self.id = id
        if self.flag_2 and name != 'unknown'and self.id != 0:
            nowtime = datetime.datetime.now()
            self.InsertLog(nowtime)
            self.flag_2 = False
    # 摄像头帧率太低，不完善
    def blink_recognise(self, img):
        #QSystemTrayIcon.showMessage()


        #msgBox.exec()
        frame_counter = 0  # 连续帧计数
        blink_counter = 0  # 眨眼计数
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转成灰度图像
        rects = detector(gray, 0)  # 人脸检测
        for rect in rects:  # 遍历每一个人脸
            # print('-' * 20)
            shape = predictor(gray, rect)  # 检测特征点
            points = face_utils.shape_to_np(shape)  # convert the facial landmark (x, y)-coordinates to a NumPy array
            leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]  # 取出左眼对应的特征点
            rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]  # 取出右眼对应的特征点
            leftEAR = self.eye_aspect_ratio(leftEye)  # 计算左眼EAR
            rightEAR = self.eye_aspect_ratio(rightEye)  # 计算右眼EAR
            # print('leftEAR = {0}'.format(leftEAR))
            # print('rightEAR = {0}'.format(rightEAR))

            ear = (leftEAR + rightEAR) / 2.0  # 求左右眼EAR的均值

            leftEyeHull = cv2.convexHull(leftEye)  # 寻找左眼轮廓
            rightEyeHull = cv2.convexHull(rightEye)  # 寻找右眼轮廓
            cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)  # 绘制左眼轮廓
            cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)  # 绘制右眼轮廓

            # 如果EAR小于阈值，开始计算连续帧，只有连续帧计数超过EYE_AR_CONSEC_FRAMES时，才会计做一次眨眼
            if ear < EYE_AR_THRESH:
                frame_counter += 1
                blink_counter += 1
                #print(frame_counter)
            else:
                if frame_counter >= EYE_AR_CONSEC_FRAMES:
                    blink_counter += 1
                frame_counter = 0

            # 在图像上显示出眨眼次数blink_counter和EAR
            cv2.putText(img, "Blinks:{0}".format(blink_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),
                        2)
            cv2.putText(img, "please blink", (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img, "EAR:{:.2f}".format(ear), (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def eye_aspect_ratio(self, eye):
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

    def dlib_recognise(self, img):
        global name
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = detector(img_gray, 0)

        # 存储当前摄像头中捕获到的所有人脸的坐标/名字
        # The list to save the positions and names of current faces captured
        #pos_namelist = []
        name_namelist = []
        if len(faces) != 0:
            # 4. 获取当前捕获到的图像的所有人脸的特征，存储到 features_cap_arr
            # 4. Get the features captured and save into features_cap_arr
            features_cap_arr = []
            for i in range(len(faces)):
                shape = predictor(img, faces[i])
                features_cap_arr.append(facerec.compute_face_descriptor(img, shape))

            # 5. 遍历捕获到的图像中所有的人脸
            # 5. Traversal all the faces in the database
            for k in range(len(faces)):
                # print("##### camera person", k+1, "#####")
                # 让人名跟随在矩形框的下方
                # 确定人名的位置坐标
                # 先默认所有人不认识，是 unknown
                # Set the default names of faces with "unknown"
                name_namelist.append("unknown")

                # 每个捕获人脸的名字坐标 the positions of faces captured
                # pos_namelist.append(
                #     tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                # 对于某张人脸，遍历所有存储的人脸特征
                # For every faces detected, compare the faces in the database
                e_distance_list = []
                for i in range(len(features_known_arr)):
                    # 如果 person_X 数据不为空
                    if str(features_known_arr[i][0]) != '0.0':
                        # print("with person", str(i + 1), "the e distance: ", end='')
                        e_distance_tmp = self.return_euclidean_distance(features_cap_arr[k], features_known_arr[i])
                        # print(e_distance_tmp)
                        e_distance_list.append(e_distance_tmp)
                    else:
                        # 空数据 person_X
                        e_distance_list.append(999999999)
                # Find the one with minimum e distance
                similar_person_num = e_distance_list.index(min(e_distance_list))
                # print("Minimum e distance with person", int(similar_person_num)+1)

                if min(e_distance_list) < 0.4:
                    # person_list = os.listdir("faces/"+"person_"+str(int(similar_person_num)+1))
                    # #以jpg文件为模板，取最后四个字符之前的名字
                    # #name_namelist[k] = person_list[0][0:-4]
                    # #获取符号“.”之前的名字
                    # name_str = person_list[0]
                    # name_str = name_str[:name_str.index(".")]
                    # name_namelist[k] = name_str
                    # print(k)
                    # print(int(similar_person_num))
                    name_namelist[k] = face_names[int(similar_person_num)]
                    self.id = int(similar_person_num)+1
                    name = name_namelist[k]
                    # print("May be "+name_str)
                # else:
                #     print("Unknown person")

                # 矩形框
                # draw rectangle
                for kk, d in enumerate(faces):
                    # 绘制矩形框
                    cv2.rectangle(img, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (255, 0, 0), 2)
                    cv2.putText(img, name, (d.left()+6, d.bottom()-6), font, 1.0, (0, 255, 255), 1)
                print('\n')
            if self.flag_3 and name != 'unknown'and self.id != 0:
                nowtime = datetime.datetime.now()
                self.InsertLog(nowtime)
                self.flag_3 = False
            # 6. 在人脸框下面写人脸名字
            # 6. write names under rectangle
            # font = cv2.FONT_ITALIC
            # for i in range(len(faces)):
            #     cv2.putText(img, name_namelist[i], pos_namelist[i], font, 1.0, (0, 255, 255), 1)
        # print("Faces in camera now:", name_namelist, "\n")

    def return_euclidean_distance(self, feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    def landmark(self, img):
        b, g, r = cv2.split(img)  # 分离三个颜色通道
        img2 = cv2.merge([r, g, b])  # 融合三个颜色通道生成新图片
        dets = detector(img, 1)  # 使用detector进行人脸检测 dets为返回的结果
        # print("Number of faces detected: {}".format(len(dets)))  # 打印识别到的人脸个数
        # enumerate是一个Python的内置方法，用于遍历索引
        # index是序号；face是dets中取出的dlib.rectangle类的对象，包含了人脸的区域等信息
        # left()、top()、right()、bottom()都是dlib.rectangle类的方法，对应矩形四条边的位置
        for index, face in enumerate(dets):
            # print('face {}; left {}; top {}; right {}; bottom {}'.format(index, face.left(), face.top(), face.right(),
            #                                                              face.bottom()))
            # 画出人脸框
            # left = face.left()
            # top = face.top()
            # right = face.right()
            # bottom = face.bottom()
            # cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 1)
            # cv2.namedWindow(f, cv2.WINDOW_AUTOSIZE)
            # cv2.imshow(f, img)
            # dlib
            shape = predictor(img, face)  # 寻找人脸的68个标定点
            #print(shape)
            # print(shape.num_parts)
            # 遍历所有点，打印出其坐标，并用蓝色的圈表示出来
            for index, pt in enumerate(shape.parts()):
                # print('Part {}: {}'.format(index, pt))
                pt_pos = (pt.x, pt.y)
                cv2.circle(img, pt_pos, 2, (255, 0, 0), 1)

        # face_recognition标记特征点
        # Load the jpg file into a numpy array
        # image = face_recognition.load_image_file(img)

        # Find all facial features in all the faces in the image
        face_landmarks_list = face_recognition.face_landmarks(img)
        if len(face_landmarks_list) != 0:
            for index, name in enumerate(face_landmarks_list[0]):
                # print('Part {}: {}'.format(index, pt))
                pt = face_landmarks_list[0].get(name)
                #print(index)
                for i in range(len(pt)):
                    #pt_pos = (pt[i].x, pt[i].y)
                    cv2.circle(img, pt[i], 2, (0, 0, 255), 2)

            # print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

            # Create a PIL imagedraw object so we can draw on the picture
            pil_image = Image.fromarray(img)
            d = ImageDraw.Draw(pil_image)

            for face_landmarks in face_landmarks_list:

                # # Print the location of each facial feature in this image
                # for facial_feature in face_landmarks.keys():
                #     print("The {} in this face has the following points: {}".format(facial_feature,
                #                                                                     face_landmarks[facial_feature]))

                # Let's trace out each facial feature in the image with a line!
                for facial_feature in face_landmarks.keys():
                    d.line(face_landmarks[facial_feature], width=5)
            # pil_image.show()

    def face_recognise(self, img):
        global name
        small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        #img = self.cv2ImgAddText(img, "大家好，我是星爷", 140, 60, (255, 255, 0), 20)

        # small_frame = img
        # face_recognise_path = current_path + "\faces"
        #
        # 原始一张图片的识别
        # obama_img = face_recognition.load_image_file("XiangMenghui.jpg")
        # face_names.append("XiangMenghui")
        # obama_face_encoding = face_recognition.face_encodings(obama_img)[0]
        # process_this_frame = True
        # if process_this_frame:
        #转换成rgb 格式
        new_frame = small_frame[:, :, ::-1]


        #默认hog方式
        ##face_locations = face_recognition.face_locations(new_frame)
        face_locations = face_recognition.face_locations(new_frame, number_of_times_to_upsample=2, model="hog")
        face_encodings = face_recognition.face_encodings(new_frame, face_locations)
        #print(face_encodings)
        for face_encoding in face_encodings:
            match = face_recognition.compare_faces(face_codings, face_encoding, 0.4)
            # print(match)
            for i in range(len(match)):
                if match[i]:
                    name = face_names[i]
                    self.id = i+1
                    break
                if i == len(match) - 1:
                    name = "unknown"
                # break
        # process_this_frame = not process_this_frame

        for (top, right, bottom, left) in (face_locations):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
            # cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), 2)
            cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (0, 255, 255), 1)
        if self.flag_1 and name != 'unknown' and self.id != 0:
            nowtime = datetime.datetime.now()
            self.InsertLog(nowtime)
            self.flag_1 = False
        # else:
        #     print("success")
        #QMessageBox.about(self, "提示对话框", "你的Windows系统是DOS1.0")

    def InsertLog(self,nowtime):
        # 打开数据库连接
        conn = pymysql.connect(host="", user="root",
                               password="xmh981127", db="face", port=3306)
        # 使用cursor()方法获取操作游标
        cur = conn.cursor()
        # 1.查询操作
        # 编写sql 查询语句  user 对应我的表名
        sql = "insert into faceserver_log(id_id, time) values(%s, %s)"
        try:
            cur.execute(sql, [self.id, nowtime])  # 执行sql语句
            conn.commit()
        except Exception as e:
            raise e
        finally:
            conn.close()  # 关闭连接


    def ColorAdjust(self, img):
        try:
            B = img[:, :, 0]
            G = img[:, :, 1]
            R = img[:, :, 2]
            B = B * self.B
            G = G * self.G
            R = R * self.R
            # B.astype(cv2.PARAM_UNSIGNED_INT)
            # G.astype(cv2.PARAM_UNSIGNED_INT)
            # R.astype(cv2.PARAM_UNSIGNED_INT)

            img1 = img
            img1[:, :, 0] = B
            img1[:, :, 1] = G
            img1[:, :, 2] = R
            return img1
        except Exception as e:
            self.MsgTE.setPlainText(str(e))

    def DispImg(self):
        if self.GrayImgCkB.isChecked():
            img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2RGB)
        qimg = qimage2ndarray.array2qimage(img)
        self.DispLb.setPixmap(QPixmap(qimg))
        self.DispLb.show()

    def StopCamera(self):
        if self.StopBt.text() == '暂停':
            self.StopBt.setText('继续')
            self.RecordBt.setText('保存')
            self.ShowBt.setEnabled(True)
            self.ShowBt.setText('保存到已有')
            self.Timer.stop()
        elif self.StopBt.text() == '继续':
            self.StopBt.setText('暂停')
            self.RecordBt.setText('录像')
            self.ShowBt.setEnabled(False)
            self.ShowBt.setText('开始')
            self.Timer.start(10)

    def RecordCamera(self):
        tag = self.RecordBt.text()
        if tag == '保存':
            try:
                # image_name=self.RecordPath+'image'+time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))+'.jpg'
                # print(image_name)
                face_path = "E:/pyqtt/faces/"
                if os.listdir(face_path):
                    # 获取已录入的最后一个人脸序号 / Get the num of latest person
                    person_list = os.listdir(face_path)
                    person_num_list = []
                    for person in person_list:
                        person_num_list.append(int(person.split('_')[-1]))
                    person_cnt = max(person_num_list)

                # 如果第一次存储或者没有之前录入的人脸, 按照 person_1 开始录入
                # Start from person_1
                else:
                    person_cnt = 0
                person_cnt += 1
                face_path = face_path + "person_" + str(person_cnt)
                self.MsgTE.setPlainText(face_path)
                os.makedirs(face_path)
                file_path = QFileDialog.getSaveFileName(self, "保存文件", face_path,
                                                        "jpg files (*.jpg);;all files(*.*)")
                #print(file_path[0])

                cv2.imwrite(file_path[0], self.Image)
                namestr = file_path[0]
                namestr = namestr.split('/')[-1].split('.')[0]
                #写入数据库
                # 打开数据库连接
                conn = pymysql.connect(host="", user="root",
                                       password="xmh981127", db="face", port=3306)

                # 使用cursor()方法获取操作游标
                cur = conn.cursor()
                # 1.查询操作
                # 编写sql 查询语句  user 对应我的表名
                sql = "insert into faceserver_student(id, name) values(%s, %s)"
                try:
                    cur.execute(sql, [person_cnt, namestr])  # 执行sql语句
                    conn.commit()
                except Exception as e:
                    raise e
                finally:
                    conn.close()  # 关闭连接

                #print(namestr)
                self.MsgTE.clear()
                self.MsgTE.setPlainText('Image saved.')
                # 录入人脸之后便获取特征并更新
                # os.system("python36 getfacefeatures_to_csv.py")
            except Exception as e:
                self.MsgTE.clear()
                self.MsgTE.setPlainText(str(e))
        elif tag == '录像':
            self.RecordBt.setText('停止')

            video_name = self.RecordPath + 'video' + time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())) + '.avi'
            fps = self.FmRateLCD.value()
            size = (self.Image.shape[1], self.Image.shape[0])
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            # self.video_writer = cv2.VideoWriter(video_name, fourcc,self.camera.get(5), size)
            self.video_writer = cv2.VideoWriter(video_name, fourcc, fps, size)
            self.RecordFlag = 1
            self.MsgTE.setPlainText('Video recording...')
            self.StopBt.setEnabled(False)
            self.ExitBt.setEnabled(False)
        elif tag == '停止':
            self.RecordBt.setText('录像')
            self.video_writer.release()
            self.RecordFlag = 0
            self.MsgTE.setPlainText('Video saved.')
            self.StopBt.setEnabled(True)
            self.ExitBt.setEnabled(True)

    def ExitApp(self):
        self.Timer.Stop()
        self.camera.release()
        self.MsgTE.setPlainText('Exiting the application..')
        QCoreApplication.quit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = MainWindow()
    child = childWindow()

    btn = ui.showLog
    btn.clicked.connect(child.show)

    ui.show()

    sys.exit(app.exec_())