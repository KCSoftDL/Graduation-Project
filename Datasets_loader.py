import tensorflow as tf
import numpy as np
import cv2
import os
import pathlib
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

AUTOTUNE = tf.data.experimental.AUTOTUNE
im_height = 224
im_width = 224
batch_size = 64
epochs = 1000

def preprocess_datapath(data_path,type):
    if (type == "train"):
        label_file = data_path + "/train_list.txt"
        data_path = data_path + "/train/"
    elif (type == "test"):
        label_file = data_path + "/test_truth_list.txt"
        data_path = data_path + "/test/"
    elif (type == "val"):
        label_file = data_path + "/val_list.txt"
        data_path = data_path + "/val/"

    return data_path,label_file

def data_loader(data_path, type , shuffle = True):

    data_path,label_file = preprocess_datapath(data_path,type)
    print(data_path)

    # 旧读数据集文件夹API
    # lables = os.listdir(data_path)
    # print(lables)
    # images = []
    # for label in lables:
    #     image = os.listdir(data_path + label)
    #     # print("The Type {} images have {}".format(label,len(image)))
    #     for i in range(len(image)):
    #         image[i] = label + "/" + image[i]
    #     # print(images)
    #     # images.append(image)
    #     images = images + image
    #
    # # print(images)
    # image_len = len(images)
    # print("total length:{}".format(image_len))
    #
    # print("label_names:",lables)

    data_root = pathlib.Path(data_path)
    # for item in data_root.iterdir():
    #     print(item)

    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    # print(all_image_paths)
    # random.shuffle(all_image_paths)

    image_count = len(all_image_paths)
    print(image_count)

    label_names = sorted(item.name for item in data_root.glob('*') if item.is_dir())
    print("label_names:", label_names)

    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    print("label_to_index:", label_to_index)
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in all_image_paths]
    # print("First labels indices: ", all_image_labels[:-1])

    for image, label in zip(all_image_paths[:5], all_image_labels[:5]):
        print(image, ' --->  ', label)

    # print(len(all_image_labels))
    # img_path = all_image_paths[0]
    # label = all_image_labels[0]

    # plt.imshow( load_and_preprocess_image(img_path) )
    # plt.xlabel(label)
    # plt.show()

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths).map(tf.io.read_file)

    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    print(image_ds)
    #
    # label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
    # print(label_ds)

    ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))

    # 元组被解压缩到映射函数的位置参数中
    def load_and_preprocess_from_path_label(path, label):
        return load_and_preprocess_image(path), label

    image_label_ds = ds.map(load_and_preprocess_from_path_label)
    print(image_label_ds)
    count = 0
    for item in image_label_ds:
        print("image_label_ds:")
        print("{} --> {}".format(item,item[item]))
        count = count+1
        if(count >= 1):
            break;


    # 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据被充分打乱。
    image_label_ds = image_label_ds.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))

    image_label_ds = image_label_ds.batch(batch_size = batch_size)

    image_label_ds = image_label_ds.prefetch(buffer_size=AUTOTUNE)
    print(image_label_ds)

    return image_label_ds

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)

def preprocess_image(image):
    # # cast image to float32 type
    # image = tf.cast(image, tf.float32)
    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.convert_image_dtype(image, tf.float32)   # normalize to [0,1] range 即 image /= 255.0
    # resize images
    image = tf.image.resize(image, [im_height,im_width])

    # data augmentation
    image = augment(image)

    return image

def augment(image):
    """ Image augmentation """
    image = _random_flip(image)
    return image


def _random_flip(image):
    """ Flip augmentation randomly"""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    return image

def load_test_data(data_path):
    """读取测试文件夹中的图片及对应的正确标签
    :param datapath: 文件夹地址
    :param labelpath: 标签文件(.txt）文件地址
    :returns images: 图片集
             labels: 字典--图片对应label
             imagespath
    """
    datapath,labelpath = preprocess_datapath(data_path, "test")
    imagepath = os.listdir(datapath)

    images = []
    # for i in range(len(imagepath)):
    #     imagepath[i] = datapath + imagepath[i]
    # path_ds = tf.data.Dataset.from_tensor_slices(imagepath).map(tf.io.read_file)
    #
    # image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    # print(image_ds)
    for i in range(len(imagepath)):
        # print(datapath+imagepath[i])
        images.append( load_and_preprocess_image(datapath+imagepath[i]))
    images = tf.stack(images, axis=0)
    print(images)

    labels = open(labelpath,"r",encoding='utf-8').readlines()
    labels = [line.split(' ') for line in labels]
    labels = dict(labels)
    # print(labels)
    return images,labels,imagepath



def load_data_by_keras(data_path,type,shuffle= False):

    data_path, label_file = preprocess_datapath(data_path, type)
    print(data_path)

    lables = os.listdir(data_path)
    print("the class is {}".format(lables))

    train_image_generator = ImageDataGenerator(rescale=1. / 255,  # 归一化
                                               rotation_range=40,  # 旋转范围
                                               width_shift_range=0.1,  # 水平平移范围
                                               height_shift_range=0.1,  # 垂直平移范围
                                               shear_range=0.1,  # 剪切变换的程度
                                               zoom_range=0.1,  # 缩放范围
                                               horizontal_flip=True,  # 水平翻转
                                               fill_mode='nearest')
    # 使用图像生成器从文件夹train_dir中读取样本，对标签进行one-hot编码
    train_data_gen = train_image_generator.flow_from_directory(directory=data_path,
                                                               batch_size=batch_size,
                                                               shuffle=shuffle,  # 打乱数据
                                                               target_size=(im_height, im_width),
                                                               class_mode='categorical')
    total_train = train_data_gen.n
    # print(total_train)

    data, labels = train_data_gen.next()
    print(labels)
    print(len(labels))
    plt.imshow(data[16, :, :, :])
    plt.title(labels[16][0])
    plt.show()

    # num = 0
    # for i in range(len(labels)):
    #     num = num + len(labels[i])
    # print(num)

    return train_data_gen

if __name__ == "__main__":
    filepath = "D:\BaiduNetdiskDownload\dataset_release/release_data"

    train_data = data_loader(filepath,type="train",shuffle= True)
    # val_data = data_loader(filepath,type="val",shuffle=True)
    # val_data = load_data_by_keras(filepath,type="val",shuffle=True)

    # test_images,true_labels,test_imagepath =load_test_data(filepath)
    # print(true_labels[test_imagepath[1]])
