# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ctg.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas 
from matplotlib.figure import Figure
class UI(QMainWindow):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setStyleSheet("\n"
"\n"
"QWidget {\n"
"    background-color: #2E2E2E;\n"
"    color: #FFFFFF;\n"
"    font-family: Arial, sans-serif;\n"
"    font-size: 14px;\n"
"}\n"
"\n"
"\n"
"QTableWidget {\n"
"    background-color: #1E1E1E;\n"
"    color: #FFFFFF;\n"
"    gridline-color: #444444;\n"
"    font-family: Arial, sans-serif;\n"
"    font-size: 14px;\n"
"}\n"
"\n"
"QTableWidget::item {\n"
"    border: 1px solid #444444;\n"
"    padding: 5px;\n"
"}\n"
"\n"
"QTableWidget::item:selected {\n"
"    background-color: #FF8C00; /* Bright orange for selected items */\n"
"    color: #FFFFFF;\n"
"}\n"
"\n"
"QHeaderView::section {\n"
"    background-color: #3A3A3A;\n"
"    color: #FFFFFF;\n"
"    padding: 5px;\n"
"    border: 1px solid #444444;\n"
"}\n"
"\n"
"\n"
"QPushButton {\n"
"    background-color: #3A3A3A;\n"
"    color: #FFFFFF;\n"
"    border: 1px solid #FF8C00; /* Bright orange border */\n"
"    border-radius: 5px;\n"
"    padding: 5px 10px;\n"
"    font-family: Arial, sans-serif;\n"
"    font-size: 14px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #FF8C00; /* Bright orange on hover */\n"
"    color: #FFFFFF;\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: #2A2A2A;\n"
"    color: #FFFFFF;\n"
"}\n"
"\n"
"\n"
"\n"
"")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.upload = QtWidgets.QPushButton(self.centralwidget)
        self.upload.setObjectName("upload")
        self.horizontalLayout.addWidget(self.upload)
        self.analyse = QtWidgets.QPushButton(self.centralwidget)
        self.analyse.setObjectName("analyse")
        self.horizontalLayout.addWidget(self.analyse)
        self.diagnose = QtWidgets.QPushButton(self.centralwidget)
        self.diagnose.setObjectName("diagnose")
        self.horizontalLayout.addWidget(self.diagnose)
        self.save = QtWidgets.QPushButton(self.centralwidget)
        self.save.setObjectName("save")
        self.horizontalLayout.addWidget(self.save)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.draw_widget = FigureCanvas(Figure(figsize=(5, 3)))
        self.draw_widget.setObjectName("draw_widget")
        self.verticalLayout.addWidget(self.draw_widget)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.table_widget = QtWidgets.QTableWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.table_widget.sizePolicy().hasHeightForWidth())
        self.table_widget.setSizePolicy(sizePolicy)
        self.table_widget.setObjectName("table_widget")
        self.table_widget.setColumnCount(0)
        self.table_widget.setRowCount(0)
        self.gridLayout.addWidget(self.table_widget, 1, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
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
        self.upload.setText(_translate("MainWindow", "Upload"))
        self.analyse.setText(_translate("MainWindow", "Analyse"))
        self.diagnose.setText(_translate("MainWindow", "Diagnose"))
        self.save.setText(_translate("MainWindow", "Save"))
