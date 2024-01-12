from pysrc.user_interface.MainMenu import Ui_MainWindow, QtCore, QtGui, QtWidgets

from PyQt5.QtWidgets import QMainWindow


class SecondWindow(QMainWindow):
    def __init__(self):
        super(SecondWindow, self).__init__()
        self.ui.setupUi()


class SGEA_MainWindow(Ui_MainWindow):
    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)

        self.radioButton_DataCollecting.clicked.connect(self.click_menu1_data_collecting)
        self.radioButton_PostProcessing.clicked.connect(self.click_menu1_post_processing)
        self.radioButton_ErrorAssessment.clicked.connect(self.click_menu1_error_assessment)

    def click_menu1_data_collecting(self):
        self.stackedWidget.setCurrentIndex(0)

    def click_menu1_post_processing(self):
        self.stackedWidget.setCurrentIndex(1)

    def click_menu1_error_assessment(self):
        self.stackedWidget.setCurrentIndex(2)
