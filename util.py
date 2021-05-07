import os
import pathlib
import pandas as pd
import tensorflow as tf
from tflearn.data_utils import build_hdf5_image_dataset

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
            image = os.listdir(path + "/"+ lable)
            for i in range(len(image)):
                image[i] = lable + "/" + image[i]
                f.write(image[i]+'\n')


def rewrite_data(train_path,val_path):
    build_hdf5_image_dataset(train_path, image_shape=(488, 488), mode='file', output_path='new_train_488.h5',
                             categorical_labels=True, normalize=False)
    build_hdf5_image_dataset(val_path, image_shape=(224, 224), mode='file', output_path='new_val_224.h5',
                             categorical_labels=True, normalize=False)

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

if __name__ == "__main__":
    path = "D:\BaiduNetdiskDownload\dataset_release/release_data"
    txt_path = path + "/trainval.txt"
    text_path ="D:\健康拍立得\datasets\ChineseFoodNet/useless.txt"
    # train_path = write_txt_from_dir(path + '/train','train',target_size=448)
    # val_path = write_txt_from_dir(path,'val',target_size=224)
    # read_chinesefoodnet_from_xlsx(path)
    # train_path = os.path.join(path,'train')
    # create_VOC_data_txt(train_path,txt_path)
    test_path = "D:\data"
    # read_and_rewrite(text_path)
    remove_useless_img(test_path,text_path)

