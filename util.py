import os
import pathlib
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

def rewrite_data(train_path,val_path):
    build_hdf5_image_dataset(train_path, image_shape=(488, 488), mode='file', output_path='new_train_488.h5',
                             categorical_labels=True, normalize=False)
    build_hdf5_image_dataset(val_path, image_shape=(224, 224), mode='file', output_path='new_val_224.h5',
                             categorical_labels=True, normalize=False)

if __name__ == "__main__":
    path = "D:\Programming/tensorflow\data/train"
    train_path = write_txt_from_dir(path,'train',target_size=448)
    val_path = write_txt_from_dir(path,'val',target_size=224)

