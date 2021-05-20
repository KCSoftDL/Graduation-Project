import os
import pathlib
import pandas as pd
import tensorflow as tf
from tflearn.data_utils import build_hdf5_image_dataset
import xml.etree.ElementTree as ET

def read_from_dir_and_write(path):
    data_root = pathlib.Path(path)
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    # print(all_image_paths)

    textpath = path +'/useless.txt'
    with open(textpath,'w+',encoding='utf-8') as f:
        for obj in all_image_paths:
            print("write '{}' into txt".format(obj))
            obj = obj + '\n'
            f.write(obj)
    print('finish write,thanks!')

def read_and_rewrite(path):
    '''
    处理useless.txt文本中的前缀，并写下处理结果
    :param path:
    :return:
    '''
    a = "F:"
    b = "useless"
    with open(path,'r+',encoding='utf-8') as f:
        filenames = f.readlines()
        for filename in filenames:
            print(filename)
            # filename.replace(b,"",1)
            # print(filename)
            list_str = list(filename)
            for i in range(11):
                list_str.pop(0)
            filename =''.join(list_str)
            print(filename)
            f.writelines(filename)
        # filename.lstrip(b)
        # print(filename)

def remove_useless_img(imgpath,txtpath):
    '''
    删除imgpath路径下,txtpath文件中记录的内容
    :param imgpath: 图像路径
    :param txtpath: 文本路径
    :return:
    '''
    with open(txtpath, 'r+', encoding='utf-8') as f:
        filenames = f.readlines()
        for filename in filenames:
            filename = filename.rstrip('\n')
            # print(filename)
            img = os.path.join(imgpath,filename)
            print("now filename is {}".format(img))
            if os.path.exists(img):
                print("remove {}".format(img))
                os.remove(img)
            else:print("文件不存在，pass!")
    print("remove all right!")

def write_txt_from_dir(path,name,target_size):
    train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,  # 归一化
                                               rotation_range=40,  # 旋转范围
                                               width_shift_range=0.1,  # 水平平移范围
                                               height_shift_range=0.1,  # 垂直平移范围
                                               shear_range=0.1,  # 剪切变换的程度
                                               zoom_range=0.1,  # 缩放范围
                                               horizontal_flip=True,  # 水平翻转
                                               fill_mode='nearest')
    train_image_reader = tf.keras.preprocessing.image.DirectoryIterator(path,train_image_generator,target_size=(target_size,target_size),shuffle= True)
    print(train_image_reader.labels)
    # print(train_image_reader.filepaths)
    images = train_image_reader.filepaths
    labels = train_image_reader.labels
    textpath = path +'/'+ name + '.txt'
    with open(textpath, 'w+', encoding='utf-8') as f:
        for i in range(len(train_image_reader.filepaths)):
            obj = images[i] + " " + str(labels[i]) + '\n'
            f.write(obj)
    return textpath

def create_VOC_data_txt(path,txt_path):
    lables = os.listdir(path)
    with open(txt_path,'w+', encoding='utf-8') as f:
        for lable in lables:
            print("start write {}'s image data".format(lable))
            image = os.listdir(path + "/"+ lable)
            for i in range(len(image)):
                image[i] = lable + "/" + image[i]
                write_txt = image[i].replace(".jpg","")
                print("now,write the {}".format(write_txt))
                f.write(write_txt+'\n')
    print("Finish writing!")

def rewrite_data(train_path,val_path):
    build_hdf5_image_dataset(train_path, image_shape=(488, 488), mode='file', output_path='new_train_488.h5',
                             categorical_labels=True, normalize=False)
    print('Done creating new_train_488.h5')
    build_hdf5_image_dataset(val_path, image_shape=(224, 224), mode='file', output_path='new_val_224.h5',
                             categorical_labels=True, normalize=False)
    print('Done creating new_val.h5')

def read_chinesefoodnet_from_xlsx(path):
    '''
    读取excle的文件前三列
    :param path: the root of .xls
    :return: 3 list : id,ChineseName,EnglishName
    '''
    path = os.path.join(path,"class_names.xls")
    # print(path)
    cols0 = pd.read_excel(path,sheet_name='Sheet1',usecols=[0]).values
    cols1 = pd.read_excel(path, sheet_name='Sheet1', usecols=[1]).values
    cols2 = pd.read_excel(path, sheet_name='Sheet1', usecols=[2]).values
    # print("{}-->{}/{}".format(cols0[0],cols1[0],cols2[0]))
    # print(cols1[000][0])
    return cols0,cols1,cols2

def rewrite_xml(path,new_path):
    '''
    替换错误的标签名，然后按正确的命名方式保存Xml标注文件
    :param path: 原文件存放path
    :param new_path: 新文件存放Path
    :return:
    '''
    labels = os.listdir(path)
    useless = os.path.join(new_path,"useless.txt")
    f = open(useless,'w+',encoding='utf-8')
    for label in labels:
        print("now write {} files".format(label))
        cwd = os.path.join(path,label)
        files = os.listdir(cwd)
        print(files)
        old_file = files[0]
        for file in files:
            t = 0
            print("read {} file".format(file))
            if(file == 'desktop.ini'):
                continue
            filename = os.path.join(cwd,file)
            tree = ET.parse(filename)
            objs = tree.findall('object')
            for obj in objs:
                obj.find('name').text = label
            file = file.replace(".xml","")
            # print(file)

            #验证是否连续，不连续写下当前文件
            index = int(file)
            if ( old_file == files[0]):
                old = int(file)
                t = 1
            old_file = file
            if  (not index == old +1 and t):
                f.writelines(label+'/'+str(old+1)+'\n')
                while( not index == old):
                    old += 1
                    if not (index == old):
                        f.writelines(label + '/' + str(old) + '\n')


            file_path = os.path.join(new_path,label)
            if not (os.path.exists(file_path)):
                os.mkdir(file_path)
            file_path = os.path.join(file_path,file + ".xml")
            print("write in {}".format(file_path))
            tree.write(file_path, encoding="utf-8",xml_declaration=True)

    f.close()
    print("Finish writing,all xml are right!")

import csv
def temp():
    excle_file = "B-CNN_train.csv"
    excle_path = os.path.join(os.getcwd(),excle_file)
    f = open(excle_path,'w',encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["train_loss", "train_acc"])

    csv_writer.writerow(["l", '18'])

    f.close()


from xml.etree.ElementTree import ElementTree, Element
import cv2


def read_xml(in_path):
    '''''读取并解析xml文件
     in_path: xml路径
     return: ElementTree
     '''
    tree = ElementTree()
    tree.parse(in_path)
    return tree


def search_jpeg():
    url = "D:\BaiduNetdiskDownload\dataset_release/release_data"
    xml_path ="D:\健康拍立得\datasets\ChineseFoodNet/label"
    url = os.path.join(url,'train')
    labels = os.listdir(url)
    for label in labels:
        path = os.path.join(url,label)
        xml = os.path.join(xml_path,label)
        print("search {}".format(label))
        for item in os.listdir(path):
            width = cv2.imread(path + "/" + item).shape[0]
            file = item.replace(".jpg",".xml")
            file = os.path.join(xml,file)
            tree = read_xml(file)
            objs = tree.findall('object')
            for obj in objs:
                bbox = obj.find('bndbox')
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                if width < y2:  # 这里因为我知道出问题的图片width为1920，所以判断一下就行。如果不知道的话，需要与xml的xmax进行比较
                    print("width={},ymax={}".format(width,x2))
                    print(label + "/"+ item)

if __name__ == "__main__":
    path = "D:\BaiduNetdiskDownload\dataset_release/release_data"
    txt_path = path + "/trainval.txt"
    text_path =path +"/useless.txt"
    test_path = "D:\健康拍立得\datasets\ChineseFoodNet"
    # train_path = write_txt_from_dir(path + '/train','train',target_size=448)
    # val_path = write_txt_from_dir(path,'val',target_size=224)
    # train_path = "D:/BaiduNetdiskDownload/dataset_release/release_data/train_list.txt"
    # val_path = "D:/BaiduNetdiskDownload/dataset_release/release_data/val_list.txt"
    # rewrite_data(train_path,val_path)
    # read_chinesefoodnet_from_xlsx(path)
    # train_path = os.path.join(path,'train')
    #
    # # read_and_rewrite(text_path)
    # # remove_useless_img(train_path,text_path)
    #
    # # newpath = os.path.join(test_path,'rewrite_test')
    # # filespath = os.path.join(test_path,'error')
    # # rewrite_xml(filespath,newpath)
    #
    # create_VOC_data_txt(train_path,txt_path)
    search_jpeg()