# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI_new.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 800)
        MainWindow.setStyleSheet("#MainWindow{border-image:url(./background.jpg);}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.Panel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame.setObjectName("frame")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.SettingsFm = QtWidgets.QFrame(self.frame)
        self.SettingsFm.setMinimumSize(QtCore.QSize(400, 0))
        self.SettingsFm.setMaximumSize(QtCore.QSize(400, 16777215))
        self.SettingsFm.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.SettingsFm.setFrameShadow(QtWidgets.QFrame.Raised)
        self.SettingsFm.setObjectName("SettingsFm")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.SettingsFm)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.frame_5 = QtWidgets.QFrame(self.SettingsFm)
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.frame_5.setStyleSheet("#frame_5{border-image:url(./back2.png);}")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.frame_5)
        self.gridLayout_6.setContentsMargins(1, 1, 1, 1)
        self.gridLayout_6.setHorizontalSpacing(1)
        self.gridLayout_6.setVerticalSpacing(2)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.BrightLb = QtWidgets.QLabel(self.frame_5)
        self.BrightLb.setStyleSheet("font: 11pt \"Adobe 黑体 Std R\";")
        self.BrightLb.setObjectName("BrightLb")
        self.gridLayout_6.addWidget(self.BrightLb, 3, 0, 1, 1, QtCore.Qt.AlignLeft)
        self.ContrastSld = QtWidgets.QSlider(self.frame_5)
        self.ContrastSld.setMinimum(0)
        self.ContrastSld.setMaximum(100)
        self.ContrastSld.setOrientation(QtCore.Qt.Horizontal)
        self.ContrastSld.setObjectName("ContrastSld")
        self.gridLayout_6.addWidget(self.ContrastSld, 4, 2, 1, 1)
        self.ExpTimeSpB = QtWidgets.QSpinBox(self.frame_5)
        self.ExpTimeSpB.setMaximumSize(QtCore.QSize(60, 30))
        self.ExpTimeSpB.setMinimum(-10)
        self.ExpTimeSpB.setMaximum(-2)
        self.ExpTimeSpB.setObjectName("ExpTimeSpB")
        self.gridLayout_6.addWidget(self.ExpTimeSpB, 1, 3, 1, 1)
        self.ContrastSpB = QtWidgets.QDoubleSpinBox(self.frame_5)
        self.ContrastSpB.setMaximumSize(QtCore.QSize(60, 30))
        self.ContrastSpB.setMinimum(0.0)
        self.ContrastSpB.setMaximum(100.0)
        self.ContrastSpB.setSingleStep(0.01)
        self.ContrastSpB.setObjectName("ContrastSpB")
        self.gridLayout_6.addWidget(self.ContrastSpB, 4, 3, 1, 1)
        self.ExpTimeSld = QtWidgets.QSlider(self.frame_5)
        self.ExpTimeSld.setMinimum(-10)
        self.ExpTimeSld.setMaximum(-2)
        self.ExpTimeSld.setOrientation(QtCore.Qt.Horizontal)
        self.ExpTimeSld.setObjectName("ExpTimeSld")
        self.gridLayout_6.addWidget(self.ExpTimeSld, 1, 2, 1, 1)
        self.BrightSld = QtWidgets.QSlider(self.frame_5)
        self.BrightSld.setMinimum(-64)
        self.BrightSld.setMaximum(64)
        self.BrightSld.setOrientation(QtCore.Qt.Horizontal)
        self.BrightSld.setObjectName("BrightSld")
        self.gridLayout_6.addWidget(self.BrightSld, 3, 2, 1, 1)
        self.ContrastLb = QtWidgets.QLabel(self.frame_5)
        self.ContrastLb.setStyleSheet("font: 11pt \"Adobe 黑体 Std R\";")
        self.ContrastLb.setObjectName("ContrastLb")
        self.gridLayout_6.addWidget(self.ContrastLb, 4, 0, 1, 1, QtCore.Qt.AlignLeft)
        self.GainSld = QtWidgets.QSlider(self.frame_5)
        self.GainSld.setMaximum(128)
        self.GainSld.setOrientation(QtCore.Qt.Horizontal)
        self.GainSld.setObjectName("GainSld")
        self.gridLayout_6.addWidget(self.GainSld, 2, 2, 1, 1)
        self.GainLb = QtWidgets.QLabel(self.frame_5)
        self.GainLb.setStyleSheet("font: 11pt \"Adobe 黑体 Std R\";")
        self.GainLb.setObjectName("GainLb")
        self.gridLayout_6.addWidget(self.GainLb, 2, 0, 1, 1, QtCore.Qt.AlignLeft)
        self.ExpTimeLb = QtWidgets.QLabel(self.frame_5)
        self.ExpTimeLb.setStyleSheet("font: 11pt \"Adobe 黑体 Std R\";")
        self.ExpTimeLb.setObjectName("ExpTimeLb")
        self.gridLayout_6.addWidget(self.ExpTimeLb, 1, 0, 1, 1, QtCore.Qt.AlignLeft)
        self.GainSpB = QtWidgets.QDoubleSpinBox(self.frame_5)
        self.GainSpB.setMaximumSize(QtCore.QSize(60, 30))
        self.GainSpB.setMaximum(128.0)
        self.GainSpB.setObjectName("GainSpB")
        self.gridLayout_6.addWidget(self.GainSpB, 2, 3, 1, 1)
        self.BrightSpB = QtWidgets.QDoubleSpinBox(self.frame_5)
        self.BrightSpB.setMaximumSize(QtCore.QSize(60, 30))
        self.BrightSpB.setMinimum(-64.0)
        self.BrightSpB.setMaximum(64.0)
        self.BrightSpB.setSingleStep(0.01)
        self.BrightSpB.setObjectName("BrightSpB")
        self.gridLayout_6.addWidget(self.BrightSpB, 3, 3, 1, 1)
        self.FilePathLb = QtWidgets.QLabel(self.frame_5)
        self.FilePathLb.setStyleSheet("font: 11pt \"Adobe 黑体 Std R\";")
        self.FilePathLb.setObjectName("FilePathLb")
        self.gridLayout_6.addWidget(self.FilePathLb, 6, 0, 1, 1)
        self.ColorFm = QtWidgets.QFrame(self.frame_5)
        self.ColorFm.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.ColorFm.setFrameShadow(QtWidgets.QFrame.Raised)
        self.ColorFm.setObjectName("ColorFm")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.ColorFm)
        self.gridLayout_7.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_7.setSpacing(1)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.RedColorSld = QtWidgets.QSlider(self.ColorFm)
        self.RedColorSld.setMaximum(255)
        self.RedColorSld.setProperty("value", 255)
        self.RedColorSld.setOrientation(QtCore.Qt.Horizontal)
        self.RedColorSld.setObjectName("RedColorSld")
        self.gridLayout_7.addWidget(self.RedColorSld, 0, 2, 1, 1)
        self.RedColorLb = QtWidgets.QLabel(self.ColorFm)
        self.RedColorLb.setMinimumSize(QtCore.QSize(40, 0))
        self.RedColorLb.setStyleSheet("font: 75 12pt \"Times New Roman\";\n"
"color: rgb(255, 0, 0);")
        self.RedColorLb.setObjectName("RedColorLb")
        self.gridLayout_7.addWidget(self.RedColorLb, 0, 0, 1, 1)
        self.BlueColorLb = QtWidgets.QLabel(self.ColorFm)
        self.BlueColorLb.setStyleSheet("font: 75 12pt \"Times New Roman\";\n"
"color: rgb(0, 0, 255);")
        self.BlueColorLb.setObjectName("BlueColorLb")
        self.gridLayout_7.addWidget(self.BlueColorLb, 2, 0, 1, 1)
        self.GreenColorSld = QtWidgets.QSlider(self.ColorFm)
        self.GreenColorSld.setMaximum(255)
        self.GreenColorSld.setProperty("value", 255)
        self.GreenColorSld.setOrientation(QtCore.Qt.Horizontal)
        self.GreenColorSld.setObjectName("GreenColorSld")
        self.gridLayout_7.addWidget(self.GreenColorSld, 1, 2, 1, 1)
        self.RedColorSpB = QtWidgets.QSpinBox(self.ColorFm)
        self.RedColorSpB.setMaximum(255)
        self.RedColorSpB.setProperty("value", 255)
        self.RedColorSpB.setObjectName("RedColorSpB")
        self.gridLayout_7.addWidget(self.RedColorSpB, 0, 3, 1, 1)
        self.BlueColorSld = QtWidgets.QSlider(self.ColorFm)
        self.BlueColorSld.setMaximum(255)
        self.BlueColorSld.setProperty("value", 255)
        self.BlueColorSld.setOrientation(QtCore.Qt.Horizontal)
        self.BlueColorSld.setObjectName("BlueColorSld")
        self.gridLayout_7.addWidget(self.BlueColorSld, 2, 2, 1, 1)
        self.GreenColorLb = QtWidgets.QLabel(self.ColorFm)
        self.GreenColorLb.setStyleSheet("font: 75 12pt \"Times New Roman\";\n"
"color: rgb(0, 255, 0);")
        self.GreenColorLb.setObjectName("GreenColorLb")
        self.gridLayout_7.addWidget(self.GreenColorLb, 1, 0, 1, 1)
        self.GreenColorSpB = QtWidgets.QSpinBox(self.ColorFm)
        self.GreenColorSpB.setMaximum(255)
        self.GreenColorSpB.setProperty("value", 255)
        self.GreenColorSpB.setObjectName("GreenColorSpB")
        self.gridLayout_7.addWidget(self.GreenColorSpB, 1, 3, 1, 1)
        self.BlueColorSpB = QtWidgets.QSpinBox(self.ColorFm)
        self.BlueColorSpB.setMaximum(255)
        self.BlueColorSpB.setProperty("value", 255)
        self.BlueColorSpB.setObjectName("BlueColorSpB")
        self.gridLayout_7.addWidget(self.BlueColorSpB, 2, 3, 1, 1)
        self.gridLayout_6.addWidget(self.ColorFm, 0, 0, 1, 4)
        self.FilePathLE = QtWidgets.QLineEdit(self.frame_5)
        self.FilePathLE.setReadOnly(True)
        self.FilePathLE.setObjectName("FilePathLE")
        self.gridLayout_6.addWidget(self.FilePathLE, 6, 2, 1, 1)
        self.FilePathBt = QtWidgets.QPushButton(self.frame_5)
        self.FilePathBt.setMaximumSize(QtCore.QSize(60, 30))
        self.FilePathBt.setStyleSheet("font: 11pt \"Adobe 黑体 Std R\";")
        self.FilePathBt.setObjectName("FilePathBt")
        self.gridLayout_6.addWidget(self.FilePathBt, 6, 3, 1, 1)
        self.gridLayout_5.addWidget(self.frame_5, 0, 1, 1, 3)
        self.EmptyFm = QtWidgets.QFrame(self.SettingsFm)
        self.EmptyFm.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.EmptyFm.setFrameShadow(QtWidgets.QFrame.Raised)
        self.EmptyFm.setObjectName("EmptyFm")
        self.checkBox = QtWidgets.QCheckBox(self.EmptyFm)
        self.checkBox.setGeometry(QtCore.QRect(90, 50, 211, 31))
        self.checkBox.setObjectName("checkBox")
        self.checkBox_2 = QtWidgets.QCheckBox(self.EmptyFm)
        self.checkBox_2.setGeometry(QtCore.QRect(90, 170, 101, 31))
        self.checkBox_2.setObjectName("checkBox_2")
        self.checkBox_3 = QtWidgets.QCheckBox(self.EmptyFm)
        self.checkBox_3.setGeometry(QtCore.QRect(90, 90, 211, 31))
        self.checkBox_3.setObjectName("checkBox_3")
        self.checkBox_4 = QtWidgets.QCheckBox(self.EmptyFm)
        self.checkBox_4.setGeometry(QtCore.QRect(90, 210, 101, 31))
        self.checkBox_4.setObjectName("checkBox_4")
        self.checkBox_6 = QtWidgets.QCheckBox(self.EmptyFm)
        self.checkBox_6.setGeometry(QtCore.QRect(90, 130, 211, 31))
        self.checkBox_6.setObjectName("checkBox_6")
        self.showLog = QtWidgets.QPushButton(self.EmptyFm)
        self.showLog.setGeometry(QtCore.QRect(90, 260, 75, 23))
        self.showLog.setObjectName("showLog")
        self.GrayImgCkB = QtWidgets.QCheckBox(self.EmptyFm)
        self.GrayImgCkB.setGeometry(QtCore.QRect(90, 20, 265, 20))
        self.GrayImgCkB.setStyleSheet("font: 11pt \"Adobe 黑体 Std R\";")
        self.GrayImgCkB.setObjectName("GrayImgCkB")
        self.gridLayout_5.addWidget(self.EmptyFm, 4, 1, 1, 3)
        self.gridLayout_2.addWidget(self.SettingsFm, 0, 0, 1, 1)
        self.DispFm = QtWidgets.QFrame(self.frame)
        self.DispFm.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.DispFm.setFrameShadow(QtWidgets.QFrame.Raised)
        self.DispFm.setObjectName("DispFm")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.DispFm)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.DispLb = QtWidgets.QLabel(self.DispFm)
        self.DispLb.setText("")
        pixmap = QPixmap("C:\\Users\\Administrator\\Desktop\\biye\\beijing.jpg")  # 按指定路径找到图片，注意路径必须用双引号包围，不能用单引号
        self.DispLb.setPixmap(pixmap)  # 在label上显示图片
        self.DispLb.setScaledContents(True)  # 让图片自适应label
        self.DispLb.setScaledContents(True)
        self.DispLb.setObjectName("DispLb")
        self.gridLayout_3.addWidget(self.DispLb, 1, 0, 1, 4)
        self.RecordBt = QtWidgets.QPushButton(self.DispFm)
        self.RecordBt.setObjectName("RecordBt")
        self.gridLayout_3.addWidget(self.RecordBt, 0, 2, 1, 1)
        self.StopBt = QtWidgets.QPushButton(self.DispFm)
        self.StopBt.setObjectName("StopBt")
        self.gridLayout_3.addWidget(self.StopBt, 0, 1, 1, 1)
        self.ExitBt = QtWidgets.QPushButton(self.DispFm)
        self.ExitBt.setObjectName("ExitBt")
        self.gridLayout_3.addWidget(self.ExitBt, 0, 3, 1, 1)
        self.ShowBt = QtWidgets.QPushButton(self.DispFm)
        self.ShowBt.setObjectName("ShowBt")
        self.gridLayout_3.addWidget(self.ShowBt, 0, 0, 1, 1)
        self.VideoInfoFm = QtWidgets.QFrame(self.DispFm)
        self.VideoInfoFm.setMaximumSize(QtCore.QSize(16777215, 60))
        self.VideoInfoFm.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.VideoInfoFm.setFrameShadow(QtWidgets.QFrame.Raised)
        self.VideoInfoFm.setObjectName("VideoInfoFm")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.VideoInfoFm)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.VideoInfoFm)
        self.label.setStyleSheet("font: 11pt \"Adobe 黑体 Std R\";")
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.FmRateLCD = QtWidgets.QLCDNumber(self.VideoInfoFm)
        self.FmRateLCD.setMaximumSize(QtCore.QSize(100, 16777215))
        self.FmRateLCD.setObjectName("FmRateLCD")
        self.horizontalLayout.addWidget(self.FmRateLCD)
        self.label_2 = QtWidgets.QLabel(self.VideoInfoFm)
        self.label_2.setStyleSheet("font: 11pt \"Adobe 黑体 Std R\";")
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.ImgWidthLCD = QtWidgets.QLCDNumber(self.VideoInfoFm)
        self.ImgWidthLCD.setMaximumSize(QtCore.QSize(100, 16777215))
        self.ImgWidthLCD.setObjectName("ImgWidthLCD")
        self.horizontalLayout.addWidget(self.ImgWidthLCD)
        self.ImgHeightLCD = QtWidgets.QLCDNumber(self.VideoInfoFm)
        self.ImgHeightLCD.setObjectName("ImgHeightLCD")
        self.horizontalLayout.addWidget(self.ImgHeightLCD)
        self.gridLayout_3.addWidget(self.VideoInfoFm, 4, 0, 1, 4)
        self.gridLayout_2.addWidget(self.DispFm, 0, 1, 1, 1)
        self.frame_4 = QtWidgets.QFrame(self.frame)
        self.frame_4.setMaximumSize(QtCore.QSize(16777215, 80))
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.frame_4)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.MsgTE = QtWidgets.QTextEdit(self.frame_4)
        self.MsgTE.setReadOnly(True)
        self.MsgTE.setObjectName("MsgTE")
        self.gridLayout_4.addWidget(self.MsgTE, 0, 0, 1, 1)
        self.gridLayout_2.addWidget(self.frame_4, 2, 0, 1, 2)
        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 17))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.BrightLb.setText(_translate("MainWindow", "亮度："))
        self.ContrastLb.setText(_translate("MainWindow", "对比度："))
        self.GainLb.setText(_translate("MainWindow", "增益："))
        self.ExpTimeLb.setText(_translate("MainWindow", "曝光时间："))
        self.FilePathLb.setText(_translate("MainWindow", "文件路径："))
        self.RedColorLb.setText(_translate("MainWindow", "RED"))
        self.BlueColorLb.setText(_translate("MainWindow", "BLUE"))
        self.GreenColorLb.setText(_translate("MainWindow", "GREEN"))
        self.FilePathBt.setText(_translate("MainWindow", "…"))
        self.checkBox.setText(_translate("MainWindow", "人脸识别-face_recognition"))
        self.checkBox_2.setText(_translate("MainWindow", "标记特征点"))
        self.checkBox_3.setText(_translate("MainWindow", "人脸识别-dlib"))
        self.checkBox_4.setText(_translate("MainWindow", "眨眼检测"))
        self.checkBox_6.setText(_translate("MainWindow", "人脸识别-opencv"))
        self.showLog.setText(_translate("MainWindow", "查看日志"))
        self.GrayImgCkB.setText(_translate("MainWindow", "Gray"))
        self.RecordBt.setText(_translate("MainWindow", "保存"))
        self.StopBt.setText(_translate("MainWindow", "暂停"))
        self.ExitBt.setText(_translate("MainWindow", "退出"))
        self.ShowBt.setText(_translate("MainWindow", "开始"))
        self.label.setText(_translate("MainWindow", "当前帧频："))
        self.label_2.setText(_translate("MainWindow", "图像尺寸："))
