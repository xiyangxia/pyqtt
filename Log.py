# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Log.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!
import pymysql
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QHeaderView, QTableWidgetItem
from pymysql import *

class Ui_Dialog(object):
    ids = []
    name = []
    time = []
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(640, 480)
        self.loaddate()
        self.tableWidget = QtWidgets.QTableWidget(Dialog)
        self.tableWidget.setGeometry(QtCore.QRect(0, 0, 631, 481))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setRowCount(100)

        ##设置水平表头
        self.tableWidget.setHorizontalHeaderLabels(["id", "姓名", "出入时间"])
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for i, id in enumerate(self.ids):
            self.tableWidget.setItem(i, 0, QTableWidgetItem(str(id)))

            self.tableWidget.setItem(i, 1, QTableWidgetItem(self.name[i]))
            self.tableWidget.setItem(i, 2, QTableWidgetItem(str(self.time[i])))

        # self.tableWidget.SetColLabelValue(0, "id")  # 第一列标签
        # self.tableWidget.SetColLabelValue(1, "姓名")
        # self.tableWidget.SetColLabelValue(2, "出入时间")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))


    def loaddate(self):
        self.ids = []
        self.name = []
        self.time = []
        # 打开数据库连接
        conn = pymysql.connect(host="", user="root",
                             password="xmh981127", db="face", port=3306)

        # 使用cursor()方法获取操作游标
        cur = conn.cursor()
        # 1.查询操作
        # 编写sql 查询语句  user 对应我的表名
        sql = "select log.id,name,time from student,log where student.id = log.id order by log.time desc"
        try:
            cur.execute(sql)  # 执行sql语句

            results = cur.fetchall()  # 获取查询的所有记录
            #print("id", "name", "password")
            # 遍历结果
            for row in results:
                self.ids.append(row[0])
                #print(row[0])
                self.name.append(row[1])
                self.time.append(row[2])
                # name = row[1]
                # password = row[2]
                #print(id, name, password)
        except Exception as e:
            raise e
        finally:
            conn.close()  # 关闭连接






