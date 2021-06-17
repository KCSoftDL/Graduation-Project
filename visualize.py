import os
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
# from Relation_Learning_Network import test
import imagenet_classes
import numpy as np
import util
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mainWindows import *
from VGG16 import predicts as vgg16_predict
from DenseNet import predicts as densenet_predict
from B_CNN import predict as bcnn_predict
from B_CNN import decord

class Choose_FileAndMode_Form(QWidget):
    Signal_OneParameter = pyqtSignal(str)
    def __init__(self, name='Choose_Form'):
        super(Choose_FileAndMode_Form, self).__init__()
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

        self.path = "D:\BaiduNetdiskDownload\dataset_release/release_data"


    def slot_btn_chooseFile(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(self,
                                                                "选取文件",
                                                                self.cwd,  # 起始路径
                                                                "All Files (*);;Text Files (*.txt)")  # 设置文件扩展名过滤,用双分号间隔

        if fileName_choose == "":
            print("取消选择")
            return
        else:
            self.Signal_OneParameter.emit(fileName_choose)
            print("Choose {}".format(fileName_choose))

class MessageBox(QWidget):
    def __init__(self):
        super(MessageBox,self).__init__()
        self.initUi()

    def initUi(self):
        self.setWindowTitle()

class MainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self.dialog = Choose_FileAndMode_Form('Choose')
        self.openButton.clicked.connect(self.slot_btn_chooseFile)
        self.openButton_2.clicked.connect(self.slot_btn_chooseFile_2)
        self.openButton_3.clicked.connect(self.slot_btn_chooseFile_3)

        # self.image = file
        self.path = "D:\BaiduNetdiskDownload\dataset_release/release_data"
        self.methodology = -1
        # self.lb = MyLabel(self)

        self.dialog.Signal_OneParameter.connect(self.Recfilename)

    def Recfilename(self,str):
        self.image = str
        print("in MainWindow recv{}".format(self.image))
        self.dialog.hide()
        self.predict()

    def slot_btn_chooseFile_3(self):
        print("choose file...")
        self.methodology = 3
        self.dialog.show()

    def slot_btn_chooseFile_2(self):
        print("choose file...")
        self.methodology = 2
        self.dialog.show()

    def slot_btn_chooseFile(self):
        print("choose file...")
        self.methodology = 1
        self.dialog.show()
        # print(self.image)
        # fileName_choose = os.path.join(self.path,"test/000000.jpg")


            # result = test(fileName_choose)
            # 读取imagenet_classes.py文件下记录的标签对应英文名
            # labels = imagenet_classes.get_labels()
            # print(result)

            # prediction = labels[np.argmax(result)][0]
            # print(prediction)
            # res = QMessageBox.information(self,"预测结果",prediction, QMessageBox.Yes | QMessageBox.No)
            #
            # if(QMessageBox.Yes == res):
            #     print("[info] you clicked yes button!")
            # elif (QMessageBox.No == res):
            #     print("[info] you clicked no button!")
        # print("你选择的文件为:{}".format(fileName_choose))

    def predict(self):
        # self.image = os.path.join(self.path, "test/000000.jpg")
        jpg = QtGui.QPixmap(self.image).scaled(self.showImage.width(), self.showImage.height())
        self.showImage.setPixmap(jpg)
        # id,ChineseName,EnglishName = util.read_chinesefoodnet_from_xlsx(self.path)
        if (self.methodology == 1):
            # result = 133
            # predict = vgg16_predict(self.image)
            predict = densenet_predict(self.image)
            result = predict[0]
            self.predict_result.setText("name:{} probability:{}".format(result[0][0],result[0][1]))
            self.predict_result_2.setText("name:{} probability:{}".format(result[1][0],result[1][1]))
            self.predict_result_3.setText("name:{} probability:{}".format(result[2][0], result[2][1]))
        elif(self.methodology == 2):
            # result = 123
            # self.predict_result.setText("code:{} mean:{}".format(result, ChineseName[result][0]))
            predict = bcnn_predict(self.image)
            result = predict[0]
            result = decord(self.image,result)
            self.predict_result.setText("name:{} probability:{}".format(result[0][0], result[0][1]))
            self.predict_result_2.setText("name:{} probability:{}".format(result[1][0], result[1][1]))
            self.predict_result_3.setText("name:{} probability:{}".format(result[2][0], result[2][1]))
        elif(self.methodology == 3):
            # result = 134
            img = Image.open(self.image)
            imagename = self.image.split("/")[-1]
            print(imagename)
            f = open("./data/boxes.txt",'r')
            loc = []
            boxes = f.readlines()
            # print(boxes)
            for box in boxes:
                box = box.replace('\n','')
                # print(box)
                txt = box.split(" ")
                # print(txt)
                loc.append(txt)
            f.close()
            print("loc:{}".format(loc))
            for box in loc:
                if(box[0] == imagename):
                    fig, ax = plt.subplots(1)
                    ax.imshow(img)
                    # 画矩形框
                    currentAxis = plt.gca()  # 获取当前子图
                    rect = patches.Rectangle((float(box[1]), float(box[2])), width=float(box[3]) - float(box[1]),
                                             height=float(box[4]) - float(box[2]), linewidth=2,
                                             edgecolor='r', fill=False)
                    currentAxis.add_patch(rect)
                    plt.show()
                predict = densenet_predict(self.image,type = 'crop',box= [float(box[1]),float(box[2]),float(box[3]),float(box[4])])
                result = predict[0]
                self.predict_result.setText("name:{} probability:{}".format(result[0][0], result[0][1]))
                self.predict_result_2.setText("name:{} probability:{}".format(result[1][0], result[1][1]))
                self.predict_result_3.setText("name:{} probability:{}".format(result[2][0], result[2][1]))
        else: raise ValueError

def _test():
    image= "data/test/000000.jpg"
    img = Image.open(image)
    # plt.imshow(img)
    # plt.show()
    imagename = image.split("/")[-1]
    print(imagename)
    f = open("./data/boxes.txt", 'r')
    loc = []
    boxes = f.readlines()
    # print(boxes)
    for box in boxes:
        box = box.replace('\n', '')
        # print(box)
        txt = box.split(" ")
        # print(txt)
        loc.append(txt)
    f.close()
    print("loc:{}".format(loc))
    for box in loc:
        print( box[0] == imagename )
        if (box[0] == imagename):
            # fig, ax = plt.subplots(1)
            # ax.imshow(img)
            # # 画矩形框
            # currentAxis = plt.gca()  # 获取当前子图
            # rect = patches.Rectangle((int(box[1]), int(box[2])), width= int(box[3]) - int(box[1]), height=int(box[4]) - int(box[2]), linewidth=2,
            #                          edgecolor='r', fill=False)
            # currentAxis.add_patch(rect)
            # plt.show()
            import tensorflow as tf
            # x = tf.image.crop_to_bounding_box(img,int(box[2]),int(box[1]),int(box[4])-int(box[2]),int(box[3])-int(box[1]))
            # x = tf.image.resize_with_pad(x, target_height=224, target_width=224)
            x = img.crop((int(box[1]),int(box[2]),int(box[3]),int(box[4])))
            x = x.resize((224, 224), Image.ANTIALIAS)
            plt.imshow(x)
            plt.show()
            x = tf.keras.preprocessing.image.img_to_array(x)
            x = np.expand_dims(x, axis=0)
        else: continue

if __name__=="__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())

    # test()
