import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from Datasets_loader import *
import imagenet_classes
from tensorflow.keras.preprocessing import image


def relationship_network(weight = None,
            input_tensor = None,
            input_shape = None):
    """
    关系型学习网络模型
    :param weight: 权重路径
    :param input_tensor:
    :param input_shape:
    :return: keras.model.Model
    """

    if input_tensor is None:
        img_input = keras.layers.Input(shape=input_shape)
    else:
        if not keras.backend.is_keras_tensor(input_tensor):
            img_input = keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = keras.layers.Conv2D(64,kernel_size=3,name="conv1")(img_input)
    x = keras.layers.BatchNormalization(name="conv1/bn")(x)
    x = keras.layers.Activation('relu', name='conv1/relu')(x)
    x = keras.layers.MaxPooling2D((2,2),  name='pool1')(x)

    x = keras.layers.Conv2D(64,kernel_size=3,name="conv2")(x)
    x = keras.layers.BatchNormalization(epsilon=1.001e-5,name="conv2/bn")(x)
    x = keras.layers.Activation('relu', name='conv2/relu')(x)
    x = keras.layers.MaxPooling2D((2,2),  name='pool2')(x)

    x = keras.layers.Dense(8,activation=tf.nn.relu,name='fc1')(x)
    output = keras.layers.Dense(1,activation=tf.nn.sigmoid,name='fc2')(x)

    if input_tensor is not None:
        inputs = keras.utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = keras.models.Model(inputs,output,name="relationship_network")

    # if not(weight == None or  os.path.exists(weight)  ):
    #     raise ValueError("The weight doesn't exit")

    if weight != None and os.path.exists(weight):
        model.load_weights(weight)

    return model
"""
"""
# class relationship_Network(keras.Model):
    # def network(self):


def embed_Network(weight = None,
                  include_top = True,
                  classes = 1000,
                  input_shape = (224,224,3)):
    model = keras.applications.vgg16.VGG16(
        weights = weight,#"./models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
        include_top = include_top,
        classes = classes,
        input_shape = input_shape)

    return model

def fc_layer(x,classes):
    x = keras.layers.Flatten(name='flatten')(x)
    x = keras.layers.Dense(4096, activation='relu', name='fc1')(x)
    x = keras.layers.Dense(4096, activation='relu', name='fc2')(x)
    x = keras.layers.Dense(classes, activation='softmax', name='predictions')(x)

    return x

def merge_input(x1,x2):
    print("x1 shape is {},x2 shape is {}".format(x1.shape,x2.shape))
    x = tf.concat([x1,x2],3)
    return x

def test(filepath):
    model = embed_Network(weight='imagenet',
                          include_top=True,
                          classes=1000,
                          input_shape=(224, 224, 3))
    model.summary()

    # images = []
    # images = load_and_preprocess_image(filepath)
    # images.append(image)
    # image = load_and_preprocess_image("D:\Programming/tensorflow\data\img9.jpg")
    # images.append(image)
    # plt.imshow(image)
    # plt.show()
    images = image.load_img(filepath, target_size=(224, 224))
    plt.imshow(images)
    plt.show(images)
    images = np.asarray(images)


    images = np.expand_dims(images,axis=0)

    # 读取imagenet_classes.py文件下记录的标签对应英文名
    # labels = imagenet_classes.get_labels()
    # print(labels[1])

    # outputs = model.output
    # output = fc_layer(outputs,6)
    # model = keras.Model(model.input,output)

    # model = relationship_network(input_shape=(224,224,3),weight=None)
    # model.summary()

    result = model.predict(images)
    preds = (np.argsort(result)[::-1])[0:5]
    # print(preds)

    return preds
    # print(result.shape)
    # print(labels[np.argmax(result)])

if __name__ == "__main__":
    filepath = "D:\Programming/tensorflow\data\img5.jpg"
    result = test(filepath)
    labels = imagenet_classes.get_labels()
    print(result)
    prediction = labels[np.argmax(result)]

    # for p in result:
    #     print("{} have {} '%".format(labels[p],p))
    print(prediction)

