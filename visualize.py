import os
import sys
from PyQt5.QtWidgets import *
from Relation_Learning_Network import test
import imagenet_classes
import numpy as np

class MainForm(QWidget):
    def __init__(self, name='MainForm'):
        super(MainForm, self).__init__()
        self.setWindowTitle(name)
        self.cwd = os.getcwd()  # 获取当前程序文件位置
        self.resize(300, 200)  # 设置窗体大小

        # btn 1
        self.btn_chooseFile = QPushButton(self)
        self.btn_chooseFile.setObjectName("btn_chooseFile")
        self.btn_chooseFile.setText("选取文件")

        # 设置布局
        layout = QVBoxLayout()
        layout.addWidget(self.btn_chooseFile)
        self.setLayout(layout)

        # 设置信号
        self.btn_chooseFile.clicked.connect(self.slot_btn_chooseFile)


    def slot_btn_chooseFile(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(self,
                                                                "选取文件",
                                                                self.cwd,  # 起始路径
                                                                "All Files (*);;Text Files (*.txt)")  # 设置文件扩展名过滤,用双分号间隔

        if fileName_choose == "":
            print("取消选择")
            return
        else:
            result = test(fileName_choose)
            # 读取imagenet_classes.py文件下记录的标签对应英文名
            labels = imagenet_classes.get_labels()
            print(result)

            prediction = labels[np.argmax(result)]
            print(prediction)
            res = QMessageBox.information(self,"预测结果",prediction, QMessageBox.Yes | QMessageBox.No)

            if(QMessageBox.Yes == res):
                print("[info] you clicked yes button!")
            elif (QMessageBox.No == res):
                print("[info] you clicked no button!")
        # print("你选择的文件为:{}".format(fileName_choose))

class MessageBox(QWidget):
    def __init__(self):
        super(MessageBox,self).__init__()
        self.initUi()

    def initUi(self):
        self.setWindowTitle()

if __name__=="__main__":
    app = QApplication(sys.argv)
    mainForm = MainForm('测试QFileDialog')
    mainForm.show()
    sys.exit(app.exec_())
