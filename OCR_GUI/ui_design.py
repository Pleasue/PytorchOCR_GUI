# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'DEMO_up.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1920, 1080)
        Form.setMinimumSize(QtCore.QSize(1920, 1080))
        Form.setMaximumSize(QtCore.QSize(10000, 10000))
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setGeometry(QtCore.QRect(0, -20, 1920, 80))
        self.groupBox.setMinimumSize(QtCore.QSize(1920, 80))
        self.groupBox.setMaximumSize(QtCore.QSize(1920, 80))
        self.groupBox.setObjectName("groupBox")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(0, 20, 351, 60))
        self.label_2.setMinimumSize(QtCore.QSize(0, 60))
        self.label_2.setMaximumSize(QtCore.QSize(16777215, 60))
        self.label_2.setStyleSheet("QLabel{\n"
"    background-color: rgb(255, 0, 0);\n"
"    color: rgb(255, 255, 255);\n"
"    font: 21pt \"华文细黑\";\n"
"    padding: 10px;\n"
"}")
        self.label_2.setMidLineWidth(0)
        self.label_2.setTextFormat(QtCore.Qt.AutoText)
        self.label_2.setIndent(-1)
        self.label_2.setTextInteractionFlags(QtCore.Qt.TextEditable)
        self.label_2.setObjectName("label_2")
        self.tabWidget = QtWidgets.QTabWidget(Form)
        self.tabWidget.setGeometry(QtCore.QRect(0, 59, 1432, 1004))
        self.tabWidget.setMinimumSize(QtCore.QSize(0, 1004))
        self.tabWidget.setMaximumSize(QtCore.QSize(1432, 100000))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.tabWidget.setFont(font)
        self.tabWidget.setStyleSheet("QTabWidget QTabBar::tab{\n"
"    width:140;\n"
"    height:46;\n"
"    font-family: \"黑体\";\n"
"    font-size: 16pt;\n"
"    color: rgb(255, 255, 255);\n"
"    background: rgb(43, 102, 80);\n"
"    border: 2px;\n"
"    border-color: rgb(99, 145, 182);\n"
"    border-radius: 8px;\n"
"    margin-left: 40px;\n"
"    margin-top: 6px;\n"
"    margin-right: 20px;\n"
"    margin-bottom: 16px;\n"
"}\n"
"\n"
"QTabBar::tab:hover{background:white;\n"
"    border-color: rgb(99, 145, 182);\n"
"    color:rgb(99, 145, 182);}\n"
"\n"
"QTabBar::tab:selected{\n"
"    background:white;\n"
"    color:rgb(99, 145, 182);\n"
"}\n"
"\n"
"QTabWidget {\n"
"    color: rgb(255, 255, 255);\n"
"    background:rgb(255,255,255);\n"
"}")
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.DisplayLabel = QtWidgets.QLabel(self.tab)
        self.DisplayLabel.setGeometry(QtCore.QRect(3, 5, 1422, 924))
        self.DisplayLabel.setMinimumSize(QtCore.QSize(0, 0))
        self.DisplayLabel.setMaximumSize(QtCore.QSize(1433, 933))
        self.DisplayLabel.setStyleSheet("background-color: rgb(138, 138, 138);\n"
"border-color: rgb(2, 2, 2);\n"
"font: 22pt \"黑体\";")
        self.DisplayLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.DisplayLabel.setObjectName("DisplayLabel")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.DisplayLabel_2 = QtWidgets.QLabel(self.tab_2)
        self.DisplayLabel_2.setGeometry(QtCore.QRect(3, 5, 1422, 924))
        self.DisplayLabel_2.setMinimumSize(QtCore.QSize(0, 910))
        self.DisplayLabel_2.setMaximumSize(QtCore.QSize(1432, 1004))
        self.DisplayLabel_2.setStyleSheet("background-color: rgb(138, 138, 138);\n"
"border-color: rgb(2, 2, 2);\n"
"font: 22pt \"黑体\";")
        self.DisplayLabel_2.setAlignment(QtCore.Qt.AlignCenter)
        self.DisplayLabel_2.setObjectName("DisplayLabel_2")
        self.tabWidget.addTab(self.tab_2, "")
        self.top = QtWidgets.QFrame(Form)
        self.top.setGeometry(QtCore.QRect(0, 60, 1920, 56))
        self.top.setMinimumSize(QtCore.QSize(1920, 56))
        self.top.setMaximumSize(QtCore.QSize(1920, 56))
        self.top.setStyleSheet("QFrame {\n"
"    background-color: rgb(43, 102, 80);\n"
"    border: 2px;\n"
"    border-color: rgb(0, 99, 180);\n"
"    border-radius: 10px;\n"
"}")
        self.top.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.top.setFrameShadow(QtWidgets.QFrame.Raised)
        self.top.setObjectName("top")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.top)
        self.horizontalLayout_5.setContentsMargins(1, -1, 9, -1)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        spacerItem = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem)
        self.outstate_2 = QtWidgets.QLabel(self.top)
        self.outstate_2.setMinimumSize(QtCore.QSize(0, 34))
        self.outstate_2.setSizeIncrement(QtCore.QSize(0, 34))
        self.outstate_2.setStyleSheet("QLabel{\n"
"    font: 16pt  \"黑体\";\n"
"    text-decoration: underline;\n"
"    color: white;\n"
"    background-color: rgb(43, 102, 80);\n"
"    margin: 6pt;\n"
"}")
        self.outstate_2.setObjectName("outstate_2")
        self.horizontalLayout_5.addWidget(self.outstate_2)
        spacerItem1 = QtWidgets.QSpacerItem(13, 13, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem1)
        self.light_2 = QtWidgets.QLabel(self.top)
        self.light_2.setMinimumSize(QtCore.QSize(34, 34))
        self.light_2.setMaximumSize(QtCore.QSize(34, 34))
        self.light_2.setStyleSheet("background-color: rgb(149, 149, 149);\n"
"border-radius: 16px;")
        self.light_2.setText("")
        self.light_2.setObjectName("light_2")
        self.horizontalLayout_5.addWidget(self.light_2)
        spacerItem2 = QtWidgets.QSpacerItem(20, 13, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem2)
        self.outstate = QtWidgets.QLabel(self.top)
        self.outstate.setMinimumSize(QtCore.QSize(0, 34))
        self.outstate.setSizeIncrement(QtCore.QSize(0, 34))
        self.outstate.setStyleSheet("QLabel{\n"
"    font: 16pt  \"黑体\";\n"
"    text-decoration: underline;\n"
"    color: white;\n"
"    background-color: rgb(43, 102, 80);\n"
"    margin: 6pt;\n"
"}")
        self.outstate.setObjectName("outstate")
        self.horizontalLayout_5.addWidget(self.outstate)
        spacerItem3 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem3)
        self.light = QtWidgets.QLabel(self.top)
        self.light.setMinimumSize(QtCore.QSize(34, 34))
        self.light.setMaximumSize(QtCore.QSize(34, 34))
        self.light.setStyleSheet("background-color: rgb(149, 149, 149);\n"
"border-radius: 16px;")
        self.light.setText("")
        self.light.setObjectName("light")
        self.horizontalLayout_5.addWidget(self.light)
        spacerItem4 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem4)
        self.time = QtWidgets.QLabel(self.top)
        self.time.setMinimumSize(QtCore.QSize(360, 40))
        self.time.setMaximumSize(QtCore.QSize(360, 40))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(16)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.time.setFont(font)
        self.time.setStyleSheet("QLabel{\n"
"    font: 16pt  \"黑体\";\n"
"    color: white;\n"
"    background-color: rgb(43, 102, 80);\n"
"    margin: 6pt;\n"
"}\n"
"")
        self.time.setText("")
        self.time.setObjectName("time")
        self.horizontalLayout_5.addWidget(self.time)
        self.lock_all = QtWidgets.QPushButton(self.top)
        self.lock_all.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lock_all.sizePolicy().hasHeightForWidth())
        self.lock_all.setSizePolicy(sizePolicy)
        self.lock_all.setMinimumSize(QtCore.QSize(90, 40))
        self.lock_all.setMaximumSize(QtCore.QSize(90, 40))
        self.lock_all.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.lock_all.setStyleSheet("QPushButton{\n"
"height: 46px;\n"
"width: 80px;\n"
"background:rgb(255, 255, 255);\n"
"font: 16pt \"黑体\" rgb(0, 0, 0);\n"
"border: 2px;\n"
"    border-color: rgb(136, 136, 136);\n"
"}\n"
"\n"
"QPushButton{border-radius: 8px;}\n"
"\n"
"QPushButton:hover{background:rgb(145, 145, 145);}")
        self.lock_all.setObjectName("lock_all")
        self.horizontalLayout_5.addWidget(self.lock_all)
        spacerItem5 = QtWidgets.QSpacerItem(25, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem5)
        self.admin = QtWidgets.QLabel(self.top)
        self.admin.setMinimumSize(QtCore.QSize(80, 34))
        self.admin.setMaximumSize(QtCore.QSize(80, 34))
        self.admin.setStyleSheet("QLabel{\n"
"    font: 16pt  \"黑体\";\n"
"    text-decoration: underline;\n"
"    color: white;\n"
"    background-color: rgb(43, 102, 80);\n"
"    margin: 6pt;\n"
"}")
        self.admin.setObjectName("admin")
        self.horizontalLayout_5.addWidget(self.admin)
        self.login = QtWidgets.QPushButton(self.top)
        self.login.setMinimumSize(QtCore.QSize(80, 40))
        self.login.setMaximumSize(QtCore.QSize(80, 40))
        self.login.setStyleSheet("QPushButton{\n"
"height: 40px;\n"
"width: 80px;\n"
"background:rgb(255, 255, 255);\n"
"font: 16pt \"黑体\" rgb(0, 0, 0);\n"
"border: 2px;\n"
"    border-color: rgb(136, 136, 136);\n"
"}\n"
"QPushButton{border: 2px;}\n"
"QPushButton{border-radius: 10px;}\n"
"\n"
"QPushButton:hover{background:rgb(145, 145, 145);}")
        self.login.setObjectName("login")
        self.horizontalLayout_5.addWidget(self.login)
        self.quit = QtWidgets.QPushButton(self.top)
        self.quit.setMinimumSize(QtCore.QSize(80, 40))
        self.quit.setMaximumSize(QtCore.QSize(80, 40))
        self.quit.setStyleSheet("QPushButton{\n"
"height: 40px;\n"
"width: 80px;\n"
"background:rgb(255, 255, 255);\n"
"font: 16pt \"黑体\" rgb(0, 0, 0);\n"
"border: 2px;\n"
"    border-color: rgb(136, 136, 136);\n"
"}\n"
"QPushButton{border: 2px;}\n"
"QPushButton{border-radius: 10px;}\n"
"\n"
"QPushButton:hover{background:rgb(145, 145, 145);}")
        self.quit.setObjectName("quit")
        self.horizontalLayout_5.addWidget(self.quit)
        self.toolBox = QtWidgets.QToolBox(Form)
        self.toolBox.setGeometry(QtCore.QRect(1450, 120, 460, 966))
        self.toolBox.setMinimumSize(QtCore.QSize(460, 966))
        self.toolBox.setMaximumSize(QtCore.QSize(460, 966))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(18)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.toolBox.setFont(font)
        self.toolBox.setStyleSheet("selection-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(0, 0, 0, 255), stop:1 rgba(255, 255, 255, 255));\n"
"border-color: rgb(2, 2, 2);\n"
"font: 18pt \"黑体\";")
        self.toolBox.setFrameShadow(QtWidgets.QFrame.Plain)
        self.toolBox.setObjectName("toolBox")
        self.page_1 = QtWidgets.QWidget()
        self.page_1.setGeometry(QtCore.QRect(0, 0, 460, 868))
        self.page_1.setObjectName("page_1")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.page_1)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.verticalGroupBox = QtWidgets.QGroupBox(self.page_1)
        self.verticalGroupBox.setEnabled(True)
        self.verticalGroupBox.setMinimumSize(QtCore.QSize(400, 400))
        self.verticalGroupBox.setMaximumSize(QtCore.QSize(460, 16777215))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(18)
        self.verticalGroupBox.setFont(font)
        self.verticalGroupBox.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.verticalGroupBox.setStyleSheet("font-family: \"黑体\";\n"
"font-size: 18pt;\n"
"height:50;\n"
"width:150;")
        self.verticalGroupBox.setObjectName("verticalGroupBox")
        self.trigger_set_Box = QtWidgets.QVBoxLayout(self.verticalGroupBox)
        self.trigger_set_Box.setContentsMargins(9, 9, 9, 9)
        self.trigger_set_Box.setObjectName("trigger_set_Box")
        self.trigger_ocr = QtWidgets.QHBoxLayout()
        self.trigger_ocr.setObjectName("trigger_ocr")
        self.trigger_ocr_start = QtWidgets.QPushButton(self.verticalGroupBox)
        self.trigger_ocr_start.setEnabled(False)
        self.trigger_ocr_start.setMinimumSize(QtCore.QSize(160, 45))
        self.trigger_ocr_start.setMaximumSize(QtCore.QSize(160, 45))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(16)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.trigger_ocr_start.setFont(font)
        self.trigger_ocr_start.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.trigger_ocr_start.setStyleSheet("QPushButton{\n"
"    font-family: \"黑体\";\n"
"    font-size: 16pt;\n"
"    height:50;\n"
"    width:150;\n"
"    color: rgb(255, 255, 255);\n"
"    background:rgb(58, 138, 107);\n"
"    border-color: rgb(58, 138, 107);\n"
"    border-radius: 10px;\n"
"    border: 2px;\n"
"}\n"
"\n"
"QPushButton:hover{background:rgb(149, 200, 115);}\n"
"\n"
"\n"
"QPushButton:pressed{background:rgb(149, 200, 115);}\n"
"QPushButton::disabled{background:rgb(124, 124, 124);}")
        self.trigger_ocr_start.setObjectName("trigger_ocr_start")
        self.trigger_ocr.addWidget(self.trigger_ocr_start)
        spacerItem6 = QtWidgets.QSpacerItem(21, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.trigger_ocr.addItem(spacerItem6)
        self.trigger_number_lable = QtWidgets.QLabel(self.verticalGroupBox)
        self.trigger_number_lable.setMinimumSize(QtCore.QSize(84, 56))
        self.trigger_number_lable.setMaximumSize(QtCore.QSize(84, 56))
        font = QtGui.QFont()
        font.setFamily("Sans Serif")
        font.setPointSize(16)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.trigger_number_lable.setFont(font)
        self.trigger_number_lable.setStyleSheet("font: 16pt \"Sans Serif\";")
        self.trigger_number_lable.setObjectName("trigger_number_lable")
        self.trigger_ocr.addWidget(self.trigger_number_lable)
        self.trigger_number_value = QtWidgets.QSpinBox(self.verticalGroupBox)
        self.trigger_number_value.setMinimumSize(QtCore.QSize(70, 50))
        self.trigger_number_value.setMaximumSize(QtCore.QSize(70, 50))
        self.trigger_number_value.setObjectName("trigger_number_value")
        self.trigger_ocr.addWidget(self.trigger_number_value)
        self.trigger_set_Box.addLayout(self.trigger_ocr)
        self.trigger_interval_Box = QtWidgets.QHBoxLayout()
        self.trigger_interval_Box.setObjectName("trigger_interval_Box")
        self.trigger_interval_label = QtWidgets.QLabel(self.verticalGroupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.trigger_interval_label.sizePolicy().hasHeightForWidth())
        self.trigger_interval_label.setSizePolicy(sizePolicy)
        self.trigger_interval_label.setMinimumSize(QtCore.QSize(84, 56))
        self.trigger_interval_label.setMaximumSize(QtCore.QSize(84, 56))
        font = QtGui.QFont()
        font.setFamily("Sans Serif")
        font.setPointSize(16)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.trigger_interval_label.setFont(font)
        self.trigger_interval_label.setStyleSheet("font: 16pt \"Sans Serif\";")
        self.trigger_interval_label.setObjectName("trigger_interval_label")
        self.trigger_interval_Box.addWidget(self.trigger_interval_label)
        self.trigger_interval_value = QtWidgets.QSpinBox(self.verticalGroupBox)
        self.trigger_interval_value.setMinimumSize(QtCore.QSize(70, 50))
        self.trigger_interval_value.setMaximumSize(QtCore.QSize(70, 50))
        self.trigger_interval_value.setObjectName("trigger_interval_value")
        self.trigger_interval_Box.addWidget(self.trigger_interval_value)
        spacerItem7 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.trigger_interval_Box.addItem(spacerItem7)
        self.trigger_start_time_label = QtWidgets.QLabel(self.verticalGroupBox)
        self.trigger_start_time_label.setMinimumSize(QtCore.QSize(84, 56))
        self.trigger_start_time_label.setMaximumSize(QtCore.QSize(84, 56))
        font = QtGui.QFont()
        font.setFamily("Sans Serif")
        font.setPointSize(16)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.trigger_start_time_label.setFont(font)
        self.trigger_start_time_label.setStyleSheet("font: 16pt \"Sans Serif\";")
        self.trigger_start_time_label.setObjectName("trigger_start_time_label")
        self.trigger_interval_Box.addWidget(self.trigger_start_time_label)
        self.trigger_start_time_value = QtWidgets.QSpinBox(self.verticalGroupBox)
        self.trigger_start_time_value.setMinimumSize(QtCore.QSize(70, 50))
        self.trigger_start_time_value.setMaximumSize(QtCore.QSize(70, 50))
        self.trigger_start_time_value.setObjectName("trigger_start_time_value")
        self.trigger_interval_Box.addWidget(self.trigger_start_time_value)
        self.trigger_set_Box.addLayout(self.trigger_interval_Box)
        self.trigger_save_Box = QtWidgets.QHBoxLayout()
        self.trigger_save_Box.setObjectName("trigger_save_Box")
        self.trigger_save_button = QtWidgets.QPushButton(self.verticalGroupBox)
        self.trigger_save_button.setMinimumSize(QtCore.QSize(160, 45))
        self.trigger_save_button.setMaximumSize(QtCore.QSize(160, 45))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(16)
        self.trigger_save_button.setFont(font)
        self.trigger_save_button.setStyleSheet("QPushButton{\n"
"    font-family: \"黑体\";\n"
"    font-size: 16pt;\n"
"    height:50;\n"
"    width:150;\n"
"    color: rgb(255, 255, 255);\n"
"    background:rgb(58, 138, 107);\n"
"    border-color: rgb(58, 138, 107);\n"
"    border-radius: 10px;\n"
"    border: 2px;\n"
"}\n"
"\n"
"QPushButton:hover{background:rgb(149, 200, 115);}\n"
"\n"
"\n"
"QPushButton:pressed{background:rgb(149, 200, 115);}\n"
"QPushButton::disabled{background:rgb(124, 124, 124);}")
        self.trigger_save_button.setObjectName("trigger_save_button")
        self.trigger_save_Box.addWidget(self.trigger_save_button)
        spacerItem8 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.trigger_save_Box.addItem(spacerItem8)
        self.trigger_save_list = QtWidgets.QComboBox(self.verticalGroupBox)
        self.trigger_save_list.setMinimumSize(QtCore.QSize(160, 45))
        self.trigger_save_list.setMaximumSize(QtCore.QSize(160, 45))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(16)
        self.trigger_save_list.setFont(font)
        self.trigger_save_list.setStyleSheet("QComboBox{\n"
"    font-family: \"黑体\";\n"
"    font-size: 16pt;\n"
"    height:50;\n"
"    width:120;\n"
"    color: rgb(255, 255, 255);\n"
"    background:rgb(58, 138, 107);\n"
"    border-color: rgb(58, 138, 107);\n"
"    border-radius: 10px;\n"
"    border: 2px;\n"
"}\n"
"\n"
"QComboBox:hover{background:rgb(149, 200, 115);}\n"
"\n"
"\n"
"QComboBox:pressed{background:rgb(149, 200, 115);}")
        self.trigger_save_list.setObjectName("trigger_save_list")
        self.trigger_save_list.addItem("")
        self.trigger_save_list.addItem("")
        self.trigger_save_list.addItem("")
        self.trigger_save_Box.addWidget(self.trigger_save_list)
        self.trigger_set_Box.addLayout(self.trigger_save_Box)
        self.scrollArea = QtWidgets.QScrollArea(self.verticalGroupBox)
        self.scrollArea.setMinimumSize(QtCore.QSize(344, 200))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 416, 217))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.scrollAreaWidgetContents)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.listWidget = QtWidgets.QListWidget(self.scrollAreaWidgetContents)
        self.listWidget.setEnabled(True)
        self.listWidget.setStyleSheet("font: 12pt \"Sans Serif\";")
        self.listWidget.setObjectName("listWidget")
        self.horizontalLayout_3.addWidget(self.listWidget)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.trigger_set_Box.addWidget(self.scrollArea)
        self.verticalLayout_4.addWidget(self.verticalGroupBox)
        self.rec_Box = QtWidgets.QGroupBox(self.page_1)
        self.rec_Box.setEnabled(False)
        self.rec_Box.setStyleSheet("font-family: \"黑体\";\n"
"font-size: 18pt;\n"
"height:50;\n"
"width:150;")
        self.rec_Box.setObjectName("rec_Box")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.rec_Box)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.search_box = QtWidgets.QPushButton(self.rec_Box)
        self.search_box.setMinimumSize(QtCore.QSize(160, 45))
        self.search_box.setMaximumSize(QtCore.QSize(160, 45))
        self.search_box.setStyleSheet("QPushButton{\n"
"    font-family: \"黑体\";\n"
"    font-size: 16pt;\n"
"    height:50;\n"
"    width:150;\n"
"    color: rgb(255, 255, 255);\n"
"    background:rgb(58, 138, 107);\n"
"    border-color: rgb(58, 138, 107);\n"
"    border-radius: 10px;\n"
"    border: 2px;\n"
"}\n"
"\n"
"QPushButton:hover{background:rgb(149, 200, 115);}\n"
"\n"
"\n"
"QPushButton:pressed{background:rgb(149, 200, 115);}\n"
"QPushButton::disabled{background:rgb(124, 124, 124);}")
        self.search_box.setObjectName("search_box")
        self.horizontalLayout_16.addWidget(self.search_box)
        spacerItem9 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_16.addItem(spacerItem9)
        self.stopButton2 = QtWidgets.QPushButton(self.rec_Box)
        self.stopButton2.setMinimumSize(QtCore.QSize(160, 45))
        self.stopButton2.setMaximumSize(QtCore.QSize(160, 45))
        self.stopButton2.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.stopButton2.setStyleSheet("QPushButton{\n"
"    font-family: \"黑体\";\n"
"    font-size: 16pt;\n"
"    height:50;\n"
"    width:150;\n"
"    color: rgb(255, 255, 255);\n"
"    background:rgb(58, 138, 107);\n"
"    border-color: rgb(58, 138, 107);\n"
"    border-radius: 10px;\n"
"    border: 2px;\n"
"}\n"
"\n"
"QPushButton:hover{background:rgb(149, 200, 115);}\n"
"\n"
"\n"
"QPushButton:pressed{background:rgb(149, 200, 115);}\n"
"QPushButton::disabled{background:rgb(124, 124, 124);}")
        self.stopButton2.setObjectName("stopButton2")
        self.horizontalLayout_16.addWidget(self.stopButton2)
        self.verticalLayout_6.addLayout(self.horizontalLayout_16)
        spacerItem10 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_6.addItem(spacerItem10)
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        self.save_result_button = QtWidgets.QPushButton(self.rec_Box)
        self.save_result_button.setMinimumSize(QtCore.QSize(160, 45))
        self.save_result_button.setMaximumSize(QtCore.QSize(160, 45))
        self.save_result_button.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.save_result_button.setStyleSheet("QPushButton{\n"
"    font-family: \"黑体\";\n"
"    font-size: 16pt;\n"
"    height:50;\n"
"    width:150;\n"
"    color: rgb(255, 255, 255);\n"
"    background:rgb(58, 138, 107);\n"
"    border-color: rgb(58, 138, 107);\n"
"    border-radius: 10px;\n"
"    border: 2px;\n"
"}\n"
"\n"
"QPushButton:hover{background:rgb(149, 200, 115);}\n"
"\n"
"\n"
"QPushButton:pressed{background:rgb(149, 200, 115);}\n"
"QPushButton::disabled{background:rgb(124, 124, 124);}")
        self.save_result_button.setObjectName("save_result_button")
        self.horizontalLayout_18.addWidget(self.save_result_button)
        spacerItem11 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_18.addItem(spacerItem11)
        self.close_save_button = QtWidgets.QPushButton(self.rec_Box)
        self.close_save_button.setMinimumSize(QtCore.QSize(160, 45))
        self.close_save_button.setMaximumSize(QtCore.QSize(160, 45))
        self.close_save_button.setStyleSheet("QPushButton{\n"
"    font-family: \"黑体\";\n"
"    font-size: 16pt;\n"
"    height:50;\n"
"    width:150;\n"
"    color: rgb(255, 255, 255);\n"
"    background:rgb(58, 138, 107);\n"
"    border-color: rgb(58, 138, 107);\n"
"    border-radius: 10px;\n"
"    border: 2px;\n"
"}\n"
"\n"
"QPushButton:hover{background:rgb(149, 200, 115);}\n"
"\n"
"\n"
"QPushButton:pressed{background:rgb(149, 200, 115);}\n"
"QPushButton::disabled{background:rgb(124, 124, 124);}")
        self.close_save_button.setObjectName("close_save_button")
        self.horizontalLayout_18.addWidget(self.close_save_button)
        self.verticalLayout_6.addLayout(self.horizontalLayout_18)
        self.verticalLayout_4.addWidget(self.rec_Box)
        self.control = QtWidgets.QGroupBox(self.page_1)
        self.control.setStyleSheet("font-family: \"黑体\";\n"
"font-size: 18pt;\n"
"height:50;\n"
"width:150;")
        self.control.setObjectName("control")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.control)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.open_cam = QtWidgets.QPushButton(self.control)
        self.open_cam.setEnabled(True)
        self.open_cam.setMinimumSize(QtCore.QSize(160, 45))
        self.open_cam.setMaximumSize(QtCore.QSize(160, 45))
        self.open_cam.setStyleSheet("QPushButton{\n"
"    font-family: \"黑体\";\n"
"    font-size: 16pt;\n"
"    height:50;\n"
"    width:150;\n"
"    color: rgb(255, 255, 255);\n"
"    background:rgb(58, 138, 107);\n"
"    border-color: rgb(58, 138, 107);\n"
"    border-radius: 10px;\n"
"    border: 2px;\n"
"}\n"
"\n"
"QPushButton:hover{background:rgb(149, 200, 115);}\n"
"\n"
"\n"
"QPushButton:pressed{background:rgb(149, 200, 115);}\n"
"QPushButton::disabled{background:rgb(124, 124, 124);}")
        self.open_cam.setObjectName("open_cam")
        self.horizontalLayout_6.addWidget(self.open_cam)
        spacerItem12 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem12)
        self.shut_cam = QtWidgets.QPushButton(self.control)
        self.shut_cam.setEnabled(False)
        self.shut_cam.setMinimumSize(QtCore.QSize(160, 45))
        self.shut_cam.setMaximumSize(QtCore.QSize(160, 45))
        self.shut_cam.setStyleSheet("QPushButton{\n"
"    font-family: \"黑体\";\n"
"    font-size: 16pt;\n"
"    height:50;\n"
"    width:150;\n"
"    color: rgb(255, 255, 255);\n"
"    background:rgb(58, 138, 107);\n"
"    border-color: rgb(58, 138, 107);\n"
"    border-radius: 10px;\n"
"    border: 2px;\n"
"}\n"
"\n"
"QPushButton:hover{background:rgb(149, 200, 115);}\n"
"\n"
"\n"
"QPushButton:pressed{background:rgb(149, 200, 115);}\n"
"QPushButton::disabled{background:rgb(124, 124, 124);}")
        self.shut_cam.setObjectName("shut_cam")
        self.horizontalLayout_6.addWidget(self.shut_cam)
        self.verticalLayout_2.addLayout(self.horizontalLayout_6)
        spacerItem13 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout_2.addItem(spacerItem13)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.video_start = QtWidgets.QPushButton(self.control)
        self.video_start.setEnabled(False)
        self.video_start.setMinimumSize(QtCore.QSize(160, 45))
        self.video_start.setMaximumSize(QtCore.QSize(160, 45))
        self.video_start.setStyleSheet("QPushButton{\n"
"    font-family: \"黑体\";\n"
"    font-size: 16pt;\n"
"    height:50;\n"
"    width:150;\n"
"    color: rgb(255, 255, 255);\n"
"    background:rgb(58, 138, 107);\n"
"    border-color: rgb(58, 138, 107);\n"
"    border-radius: 10px;\n"
"    border: 2px;\n"
"}\n"
"\n"
"QPushButton:hover{background:rgb(149, 200, 115);}\n"
"\n"
"\n"
"QPushButton:pressed{background:rgb(149, 200, 115);}\n"
"QPushButton::disabled{background:rgb(124, 124, 124);}")
        self.video_start.setObjectName("video_start")
        self.horizontalLayout_7.addWidget(self.video_start)
        spacerItem14 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem14)
        self.video_query = QtWidgets.QPushButton(self.control)
        self.video_query.setMinimumSize(QtCore.QSize(160, 45))
        self.video_query.setMaximumSize(QtCore.QSize(160, 45))
        self.video_query.setStyleSheet("QPushButton{\n"
"    font-family: \"黑体\";\n"
"    font-size: 16pt;\n"
"    height:50;\n"
"    width:150;\n"
"    color: rgb(255, 255, 255);\n"
"    background:rgb(58, 138, 107);\n"
"    border-color: rgb(58, 138, 107);\n"
"    border-radius: 10px;\n"
"    border: 2px;\n"
"}\n"
"\n"
"QPushButton:hover{background:rgb(149, 200, 115);}\n"
"\n"
"\n"
"QPushButton:pressed{background:rgb(149, 200, 115);}\n"
"QPushButton::disabled{background:rgb(124, 124, 124);}")
        self.video_query.setObjectName("video_query")
        self.horizontalLayout_7.addWidget(self.video_query)
        self.verticalLayout_2.addLayout(self.horizontalLayout_7)
        self.verticalLayout_4.addWidget(self.control)
        self.toolBox.addItem(self.page_1, "")
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setEnabled(True)
        self.page_2.setGeometry(QtCore.QRect(0, 0, 418, 235))
        self.page_2.setObjectName("page_2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.page_2)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.rec_Box_2 = QtWidgets.QGroupBox(self.page_2)
        self.rec_Box_2.setMinimumSize(QtCore.QSize(400, 0))
        self.rec_Box_2.setMaximumSize(QtCore.QSize(460, 16777215))
        self.rec_Box_2.setStyleSheet("font-family: \"黑体\";\n"
"font-size: 18pt;\n"
"height:50;\n"
"width:150;")
        self.rec_Box_2.setObjectName("rec_Box_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.rec_Box_2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.trigger_ocr_2 = QtWidgets.QHBoxLayout()
        self.trigger_ocr_2.setObjectName("trigger_ocr_2")
        self.open_dir = QtWidgets.QPushButton(self.rec_Box_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(160)
        sizePolicy.setVerticalStretch(45)
        sizePolicy.setHeightForWidth(self.open_dir.sizePolicy().hasHeightForWidth())
        self.open_dir.setSizePolicy(sizePolicy)
        self.open_dir.setMinimumSize(QtCore.QSize(160, 45))
        self.open_dir.setMaximumSize(QtCore.QSize(160, 45))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(16)
        self.open_dir.setFont(font)
        self.open_dir.setStyleSheet("QPushButton{\n"
"    font-family: \"黑体\";\n"
"    font-size: 16pt;\n"
"    height:50;\n"
"    width:150;\n"
"    color: rgb(255, 255, 255);\n"
"    background:rgb(58, 138, 107);\n"
"    border-color: rgb(58, 138, 107);\n"
"    border-radius: 10px;\n"
"    border: 2px;\n"
"}\n"
"\n"
"QPushButton:hover{background:rgb(149, 200, 115);}\n"
"\n"
"\n"
"QPushButton:pressed{background:rgb(149, 200, 115);}\n"
"QPushButton::disabled{background:rgb(124, 124, 124);}")
        self.open_dir.setObjectName("open_dir")
        self.trigger_ocr_2.addWidget(self.open_dir)
        spacerItem15 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.trigger_ocr_2.addItem(spacerItem15)
        self.ocr_det = QtWidgets.QPushButton(self.rec_Box_2)
        self.ocr_det.setEnabled(False)
        self.ocr_det.setMinimumSize(QtCore.QSize(160, 45))
        self.ocr_det.setMaximumSize(QtCore.QSize(160, 45))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(16)
        self.ocr_det.setFont(font)
        self.ocr_det.setStyleSheet("QPushButton{\n"
"    font-family: \"黑体\";\n"
"    font-size: 16pt;\n"
"    height:50;\n"
"    width:150;\n"
"    color: rgb(255, 255, 255);\n"
"    background:rgb(58, 138, 107);\n"
"    border-color: rgb(58, 138, 107);\n"
"    border-radius: 10px;\n"
"    border: 2px;\n"
"}\n"
"\n"
"QPushButton:hover{background:rgb(149, 200, 115);}\n"
"\n"
"\n"
"QPushButton:pressed{background:rgb(149, 200, 115);}\n"
"QPushButton::disabled{background:rgb(124, 124, 124);}")
        self.ocr_det.setObjectName("ocr_det")
        self.trigger_ocr_2.addWidget(self.ocr_det)
        self.verticalLayout.addLayout(self.trigger_ocr_2)
        self.listWidget_2 = QtWidgets.QListWidget(self.rec_Box_2)
        self.listWidget_2.setStyleSheet("font: 12pt \"Sans Serif\";")
        self.listWidget_2.setObjectName("listWidget_2")
        self.verticalLayout.addWidget(self.listWidget_2)
        self.horizontalLayout_4.addWidget(self.rec_Box_2)
        self.toolBox.addItem(self.page_2, "")
        self.toolBox.raise_()
        self.groupBox.raise_()
        self.top.raise_()
        self.tabWidget.raise_()

        self.retranslateUi(Form)
        self.tabWidget.setCurrentIndex(0)
        self.toolBox.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_2.setText(_translate("Form", "手-眼智能识别与分拣系统"))
        self.DisplayLabel.setText(_translate("Form", "实时监视区"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Form", "实时监控"))
        self.DisplayLabel_2.setText(_translate("Form", "图片显示区"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Form", "图片显示"))
        self.outstate_2.setText(_translate("Form", "待监测"))
        self.outstate.setText(_translate("Form", "状态"))
        self.lock_all.setText(_translate("Form", "LOCK"))
        self.admin.setText(_translate("Form", "管理员"))
        self.login.setText(_translate("Form", "登录"))
        self.quit.setText(_translate("Form", "退出"))
        self.verticalGroupBox.setTitle(_translate("Form", "触发设置"))
        self.trigger_ocr_start.setText(_translate("Form", "开始运行"))
        self.trigger_number_lable.setText(_translate("Form", "工件数目"))
        self.trigger_interval_label.setText(_translate("Form", "触发间隔"))
        self.trigger_start_time_label.setText(_translate("Form", "触发延迟"))
        self.trigger_save_button.setText(_translate("Form", "保存设置"))
        self.trigger_save_list.setItemText(0, _translate("Form", "触发列表1"))
        self.trigger_save_list.setItemText(1, _translate("Form", "触发列表2"))
        self.trigger_save_list.setItemText(2, _translate("Form", "触发列表3"))
        self.rec_Box.setTitle(_translate("Form", "自动识别"))
        self.search_box.setText(_translate("Form", "自动搜索"))
        self.stopButton2.setText(_translate("Form", "结束搜索"))
        self.save_result_button.setText(_translate("Form", "开始保存"))
        self.close_save_button.setText(_translate("Form", "结束保存"))
        self.control.setTitle(_translate("Form", "控制面板"))
        self.open_cam.setText(_translate("Form", "打开相机"))
        self.shut_cam.setText(_translate("Form", "关闭相机"))
        self.video_start.setText(_translate("Form", "开始录制"))
        self.video_query.setText(_translate("Form", "查看录制"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_1), _translate("Form", "相机控制"))
        self.rec_Box_2.setTitle(_translate("Form", "显示管理"))
        self.open_dir.setText(_translate("Form", "打开目录"))
        self.ocr_det.setText(_translate("Form", "单张检测"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_2), _translate("Form", "图片查阅"))