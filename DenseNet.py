import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ReduceLROnPlateau
from Datasets_loader import data_loader,load_test_data

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
    model = tf.keras.Model(base_model.inputs, predictions)
    return model

if __name__ == "__main__":

    epoch = 1000
    batch_size = 32
    learning_rate = 0.001

    filepath = "D:\BaiduNetdiskDownload\dataset_release/release_data"
    train_data = data_loader(filepath, type="train", shuffle=True)
    val_data = data_loader(filepath,type="val",shuffle=True)
    # test_images,true_labels,test_imagepath =load_test_data(filepath)

    model = keras.applications.densenet.DenseNet121(weights='./DenseNet-BC-121-32-no-top.h5',
                                                    include_top=False,
                                                    input_tensor=None,
                                                    pooling=None,
                                                    input_shape=[224,224,3])
    model = add_new_last_layer(model, 208)

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
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    history = model.fit(train_data,
                        epochs=epoch,
                        verbose = 2,
                        steps_per_epoch = 256,
                        callbacks=[TensorBoard(),reduce_lr])
    print('history dict:', history.history)

    model.save("./models/DenseNet.h5")
    print("Success Save Model!")

    model.evaluate(val_data)

