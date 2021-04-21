import time
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ReduceLROnPlateau
from Datasets_loader import data_loader,load_test_data,load_data_by_keras
from tensorflow.keras.utils import plot_model

def add_new_last_layer(base_model, nb_classes):
    '''
    添加最后的分类层
    :param base_model:
    :param nb_classes:
    :return: model
    '''
    # x = base_model.output
    x = base_model.get_layer(index=-1).output
    x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = keras.layers.Flatten(name='flatten')(x)
    predictions = keras.layers.Dense(nb_classes, activation='softmax')(x) #new softmax layer
    model = tf.keras.Model(base_model.inputs, predictions,name="DenseNet121")
    return model

def ClassificationLayer(base_model, nb_classes):
    '''
    添加分类层
    :param base_model:
    :param nb_classes:
    :return:
    '''
    global_average_layer = keras.layers.GlobalAveragePooling2D(name='avg_pool')
    flatten_layer = keras.layers.Flatten(name='flatten')
    prediction_layer = keras.layers.Dense(nb_classes, activation='softmax')
    model = keras.Sequential([base_model,
                              global_average_layer,
                              flatten_layer,
                              prediction_layer
    ],name="My_DenseNet121")
    return model

if __name__ == "__main__":

    # tf.test.is_gpu_available()
    epoch = 1000
    batch_size = 128
    learning_rate = 0.01

    filepath = "D:\datasets\ChineseFoodNet/release_data"
    # train_data,train_num = data_loader(filepath, type="train", shuffle=True)
    # val_data,v = data_loader(filepath,type="val",shuffle=True)
    # test_images,true_labels,test_imagepath =load_test_data(filepath)

    train_data,train_num = load_data_by_keras(filepath,type="train",shuffle=True)
    val_data,val_num = load_data_by_keras(filepath,type="val",shuffle=True)

    base_model = keras.applications.densenet.DenseNet121(weights='./models/DenseNet-BC-121-32-no-top.h5',
                                                    include_top=False,
                                                    input_tensor=None,
                                                    pooling=None,
                                                    input_shape=[224,224,3])
    base_model.trainable = False

    # model = add_new_last_layer(base_model, 208)
    model = ClassificationLayer(base_model,208)

    model.build((batch_size, 224, 224, 3))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         beta_1=0.9,beta_2=0.999,
                                         epsilon=1e-7)
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    train_loss = tf.keras.metrics.Mean(name="loss", dtype=tf.float32)
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    model.compile(loss=loss_object,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file='model.png')
    steps_per_epoch = round(train_num / batch_size)
    validation_steps = val_num / batch_size
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                                  patience=5, min_lr=0.001)

    model_name = "DenseNet"
    log_dir = os.getcwd()
    log_dir = os.path.join(log_dir,"logs")
    log_dir = os.path.join(log_dir,model_name)
    print("model save at {}".format(log_dir))
    tensorboard = TensorBoard(log_dir=log_dir)
    history = model.fit_generator(generator=train_data,
                        epochs=epoch,
                        verbose = 2,
                        steps_per_epoch = 128,
                        validation_data= val_data,
                        validation_steps= 128,
                        callbacks=[tensorboard,reduce_lr]
                        )
    print('history dict:', history.history)

    localtime = time.strftime("%y-%m-%d-%H:%M:%S", time.localtime())
    print(localtime)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig("./models/DenseNet_in_{}".format(localtime))
    plt.show()

    acc_text = np.array(acc)
    val_acc_text = np.array(val_acc)
    loss_text = np.array(loss)
    val_loss_text = np.array(val_loss)
    np.savetxt("./models/denseNet_acc.txt",acc_text)
    np.savetxt("./models/denseNet_val_acc.txt", val_acc_text)
    np.savetxt("./models/denseNet_loss.txt", loss_text)
    np.savetxt("./models/denseNet_val_loss.txt", val_loss_text)

    model.save("./models/DenseNet.h5")
    print("Success Save Model!")

    # model.evaluate(val_data)

