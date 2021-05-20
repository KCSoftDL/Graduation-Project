import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

image_hight = 224
image_width = 224
image_channels = 3

#不同层次的损失权重
alpha = K.variable(value=0.8, dtype="float32", name="alpha") # A1
beta = K.variable(value=0.2, dtype="float32", name="beta") # A2

#用于根据不同训练阶段调整学习率
def scheduler(epoch):
    learning_rate_init=0.0003
    if epoch>20:
        learning_rate_init=0.0005
    if epoch>30:
        learning_rate_init=0.0001
    return learning_rate_init

#用于调整不同层次的损失权重
class LossWeightsModifier(keras.callbacks.Callback):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def on_epoch_end(self, epoch, logs={}):  # focus
        if epoch == 10:
            K.set_value(self.alpha, 0.8)
            K.set_value(self.beta, 0.2)
        if epoch == 25:
            K.set_value(self.alpha, 0.1)
            K.set_value(self.beta, 0.9)
        if epoch == 40:
            K.set_value(self.alpha, 0)
            K.set_value(self.beta, 1)

"""
def bilinear_pooling(x, y):
    x_size = x.size()
    y_size = y.size()

    assert (x_size[:-1] == y_size[:-1])
    out_size = list(x_size)
    out_size[-1] = x_size[-1] * y_size[-1] # 特征x和特征y维数之积

    x = x.view([-1, x_size[-1]])  # [N*C,F]
    y = y.view([-1, y_size[-1]])

    out_stack = []

    for i in range(x.size()[0]):
        out_stack = np.cross(x[i].numpy(),y[i].numpy())
        # out_stack.append(torch.ger(x[i], y[i]))# torch.ger()向量的外积操作
        out = tf.stack(out_stack)# 将list堆叠成tensor

    return out.view(out_size)# [N,C,F*F]
"""

def VGG16_notop(img_input):
    model = keras.Sequential()

    #conv1
    model.add(layers.Conv2D(64, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block1_conv1'))
    model.add(layers.Conv2D(64, (3, 3),
                        activation='relu',
                        padding='same',
                        name='block1_conv2'))
    model.add(layers.MaxPooling2D((2, 2),
                                  strides=(2, 2),
                                  name='block1_pool'))

    #conv2
    model.add(layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1'))
    model.add(layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2'))
    model.add(layers.MaxPooling2D((2, 2),
                        strides=(2, 2),
                        name='block2_pool'))

    #conv3
    model.add(layers.Conv2D(256, (3, 3),
                    activation='relu',
                    padding='same',
                    name='block3_conv1'))
    model.add(layers.Conv2D(256, (3, 3),
                    activation='relu',
                    padding='same',
                    name='block3_conv2'))
    model.add(layers.Conv2D(256, (3, 3),
                    activation='relu',
                    padding='same',
                    name='block3_conv3'))
    model.add(layers.MaxPooling2D((2, 2),
                    strides=(2, 2),
                    name='block3_pool'))

    #conv4
    model.add(layers.Conv2D(512, (3, 3),
                    activation='relu',
                    padding='same',
                    name='block4_conv1'))
    model.add(layers.Conv2D(512, (3, 3),
                    activation='relu',
                    padding='same',
                    name='block4_conv2'))
    model.add(layers.Conv2D(512, (3, 3),
                    activation='relu',
                    padding='same',
                    name='block4_conv3'))
    model.add(layers.MaxPooling2D((2, 2),
                    strides=(2, 2),
                    name='block4_pool'))

    #conv5
    model.add(layers.Conv2D(512, (3, 3),
                    ctivation='relu',
                    padding='same',
                    name='block5_conv1'))
    model.add(layers.Conv2D(512, (3, 3),
                    activation='relu',
                    padding='same',
                    name='block5_conv2'))
    model.add(layers.Conv2D(512, (3, 3),
                    activation='relu',
                    padding='same',
                    name='block5_conv3'))
    model.add(layers.MaxPooling2D((2, 2),
                    strides=(2, 2),
                    name='block5_pool'))

    return model

def B_CNN(inputs):
    base_model = keras.applications.vgg16.VGG16(weight = './models/vgg16_weights_tf_keras_notop.h5',
                                                include_top = False,
                                                input_shape = [224,224,3])
    base_model.trainable=False

    conv5_3 = base_model.get_layer(name='block5_conv3').output

    x = layers.Flatten()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)


