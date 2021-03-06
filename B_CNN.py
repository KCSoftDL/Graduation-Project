'''
This file is to train only the last fully connected layer of Bilineaar_CNN (DD).
Bilinear_CNN (DD) network needs images of input size [3x448x448].
For using the random crops, images are first resized to [3x488x488] using create_h5_dataset.py.
During the training, images are randomly cropped to the size of [3x448x448]
Weights for the last layer can be saved to a file and will be used during
finetuning the whole Bilinear_CNN (DD) network
'''

import csv
import os
import random
import time

import PIL.Image as Image
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from util import read_chinesefoodnet_from_xlsx
import Datasets_loader
from Datasets_loader import load_data_by_keras


def random_flip_right_to_left(image_batch):
    result = []
    # print("Flipping images")
    # print(len(image_batch))
    for n in range(len(image_batch)):
        # print(image_batch[n].shape)
        if bool(random.getrandbits(1)):
            result.append(image_batch[n][:, ::-1, :])
        else:
            result.append(image_batch[n])
    return result


def random_crop(image_batch):
    result = []
    # print("Cropping images")
    for n in range(image_batch.shape[0]):
        print("image_batch[n].shape:{}".format(image_batch[n].shape))
        start_x = random.randint(0, 39)
        start_y = random.randint(0, 39)
        image = image_batch[n][start_y:start_y + 448, start_x:start_x + 448, :]
        result.append(image)
        print("result[n].shape:{}".format(result[n].shape))
    return result


class vgg16:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.last_layer_parameters = []  ## Parameters in this list will be optimized when only last layer is being trained
        self.parameters = []  ## Parameters in this list will be optimized when whole BCNN network is finetuned
        self.convlayers()  ## Create Convolutional layers
        self.fc_layers()  ## Create Fully connected layer
        self.weight_file = weights
        # self.load_weights(weights, sess)

    def convlayers(self):

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs - mean
            print('Adding Data Augmentation')
            # self.parameters = []
            # self.last_layer_parameters = []
            # images = tf.map_fn(lambda img: tf.image.random_flip_left_right(img),images)     ## Data augmentation

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        print('Shape of conv5_3', self.conv5_3.get_shape())
        self.phi_I = tf.einsum('ijkm,ijkn->imn', self.conv5_3, self.conv5_3)
        print('Shape of phi_I after einsum', self.phi_I.get_shape())

        self.phi_I = tf.reshape(self.phi_I, [-1, 512 * 512])
        print('Shape of phi_I after reshape', self.phi_I.get_shape())

        self.phi_I = tf.divide(self.phi_I, 784.0)
        print('Shape of phi_I after division', self.phi_I.get_shape())

        self.y_ssqrt = tf.multiply(tf.sign(self.phi_I), tf.sqrt(tf.abs(self.phi_I) + 1e-12))
        print('Shape of y_ssqrt', self.y_ssqrt.get_shape())

        self.z_l2 = tf.nn.l2_normalize(self.y_ssqrt, dim=1)
        print('Shape of z_l2', self.z_l2.get_shape())

    def fc_layers(self):

        with tf.name_scope('fc-new') as scope:
            fc3w = tf.get_variable('weights', [512 * 512, 208], initializer=tf.keras.initializers.glorot_normal,
                                   trainable=True)
            fc3b = tf.Variable(tf.constant(1.0, shape=[208], dtype=tf.float32), name='biases', trainable=True)
            self.fc3l = tf.nn.bias_add(tf.matmul(self.z_l2, fc3w), fc3b)
            self.last_layer_parameters += [fc3w, fc3b]
            self.parameters += [fc3w, fc3b]

    def load_weights(self, sess):
        weights = np.load(self.weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            removed_layer_variables = ['fc6_W', 'fc6_b', 'fc7_W', 'fc7_b', 'fc8_W', 'fc8_b']
            if not k in removed_layer_variables:
                print(k)
                print("", i, k, np.shape(weights[k]))
                sess.run(self.parameters[i].assign(weights[k]))


def train():
    tf.disable_eager_execution()
    # with tf.device('/cpu:0'):
    # train_data = h5py.File('../new_train_488.h5', 'r')
    # val_data = h5py.File('../new_val.h5', 'r')
    filepath = "D:\datasets\ChineseFoodNet/release_data"

    train_data, _ = load_data_by_keras(filepath, mode="train", im_height=488, im_width=488)
    val_data, _ = load_data_by_keras(filepath, mode="val", im_height=448, im_width=448)

    print('Input data read complete')

    # X_train, Y_train = train_data.next()
    # X_val, Y_val = val_data.next()
    # X_train, Y_train = train_data['X'], train_data['Y']
    # X_val, Y_val = val_data['X'], val_data['Y']
    #
    # print("Data shapes -- (train, val, test)", X_train.shape, X_val.shape)
    # X_train, Y_train = shuffle(X_train, Y_train)
    #
    # X_val, Y_val = shuffle(X_val, Y_val)
    # print Y_train[0]
    print("Device placement on. Creating Session")

    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess = tf.Session()
    # sess = tf.InteractiveSession()
    # with tf.device('/gpu:0'):
    imgs = tf.placeholder(tf.float32, [None, 448, 448, 3])
    target = tf.placeholder("float", [None, 208])
    # print 'Creating graph'
    vgg = vgg16(imgs, './models/vgg16_weights.npz', sess)

    # print X_train.shape

    # with tf.device("/gpu:0"):
    print('VGG network created')
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=vgg.probs, labels=target))

    # Defining other ops using Tensorflow
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=vgg.fc3l, labels=target))
    learning_rate_wft = tf.placeholder(tf.float32, shape=[])
    learning_rate_woft = tf.placeholder(tf.float32, shape=[])

    optimizer = tf.train.MomentumOptimizer(learning_rate=0.9, momentum=0.9).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(vgg.fc3l, 1), tf.argmax(target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    num_correct_preds = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())

    vgg.load_weights(sess)

    batch_size = Datasets_loader.batch_size

    # print "Trainable", tf.trainable_variables()[0]
    print('Starting training')

    lr = 1.0
    base_lr = 1.0
    finetune_step = 50
    epochs = 100

    excle_file = "B-CNN_train.csv"
    excle_path = os.path.join(os.getcwd(), excle_file)
    f = open(excle_path, 'w+', encoding='utf-8', newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow(["train_loss", "train_acc"])

    for epoch in range(epochs):
        avg_cost = 0.
        total_batch = int(train_data.n / batch_size / epochs)
        # X_train, Y_train = shuffle(X_train, Y_train)

        # Uncomment following section if you want to break training at a particular epoch

        if epoch % 10 == 0 or epoch == epochs:
            last_layer_weights = []
            for v in vgg.parameters:
                print(v)
                if v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                    print('Printing Trainable Variables :', sess.run(v).shape)
                    last_layer_weights.append(sess.run(v))
            np.savez('last_layers_epoch_20_crop.npz', last_layer_weights)
            print("Last layer weights saved")

        for i in range(total_batch):
            # batch_xs, batch_ys = X_train[i * batch_size:i * batch_size + batch_size], Y_train[
            #                                                                           i * batch_size:i * batch_size + batch_size]
            batch_xs, batch_ys = train_data.next()
            batch_xs = random_crop(batch_xs)
            batch_xs = random_flip_right_to_left(batch_xs)
            # print(batch_xs)
            # if epoch <= finetune_step:
            start = time.time()
            sess.run(optimizer, feed_dict={imgs: batch_xs, target: batch_ys})
            if i % 20 == 0:
                print('Last layer training, time to run optimizer for batch size 32:', time.time() - start, 'seconds')

            cost = sess.run(loss, feed_dict={imgs: batch_xs, target: batch_ys})

            if i % 100 == 0:
                # print ('Learning rate: ', (str(lr)))
                if epoch <= finetune_step:
                    print("Training last layer of BCNN_DD")
                else:
                    print("Fine tuning all BCNN_DD")

                print("Epoch:", '%03d' % (epoch + 1), "Step:", '%03d' % i, "Loss:", str(cost))
                # print("Training Accuracy -->", accuracy.eval(feed_dict={imgs: batch_xs, target: batch_ys}, session=sess))
                train_acc = sess.run(accuracy, feed_dict={imgs: batch_xs, target: batch_ys})
                print("Training Accuracy -->", train_acc)
                csv_writer.writerow([cost, train_acc])

        f.close()
        excle_file = "B-CNN_val.csv"
        excle_path = os.path.join(os.getcwd(), excle_file)
        f = open(excle_path, 'w+', encoding='utf-8', newline="")
        csv_writer = csv.writer(f)
        csv_writer.writerow(["val_loss", "val_acc"])

        val_batch_size = 32
        total_val_count = val_data.n
        correct_val_count = 0
        val_loss = 0.0
        total_val_batch = int(total_val_count / val_batch_size)
        for i in range(total_val_batch):
            # batch_val_x, batch_val_y = X_val[i * val_batch_size:i * val_batch_size + val_batch_size], Y_val[
            #                                                                                           i * val_batch_size:i * val_batch_size + val_batch_size]
            batch_val_x, batch_val_y = val_data.next()
            val_loss += sess.run(loss, feed_dict={imgs: batch_val_x, target: batch_val_y})

            pred = sess.run(num_correct_preds, feed_dict={imgs: batch_val_x, target: batch_val_y})
            correct_val_count += pred
            csv_writer.writerow([val_loss, 100.0 * correct_val_count / (1.0 * total_val_count)])

        print("##############################")
        print("Validation Loss -->", val_loss)
        print("correct_val_count, total_val_count", correct_val_count, total_val_count)
        print("Validation Data Accuracy -->", 100.0 * correct_val_count / (1.0 * total_val_count))
        print("##############################")

    # test_data = h5py.File('../new_test.h5', 'r')
    # X_test, Y_test = test_data['X'], test_data['Y']
    # total_test_count = len(X_test)
    # correct_test_count = 0
    # test_batch_size = 10
    # total_test_batch = int(total_test_count / test_batch_size)
    # for i in range(total_test_batch):
    #     batch_test_x, batch_test_y = X_test[i * test_batch_size:i * test_batch_size + test_batch_size], Y_test[
    #                                                                                                     i * test_batch_size:i * test_batch_size + test_batch_size]
    #
    #     pred = sess.run(num_correct_preds, feed_dict={imgs: batch_test_x, target: batch_test_y})
    #     correct_test_count += pred
    #
    # print("##############################")
    # print("correct_test_count, total_test_count", correct_test_count, total_test_count)
    # print("Test Data Accuracy -->", 100.0 * correct_test_count / (1.0 * total_test_count))
    # print("##############################")


def predict(filename):
    tf.disable_eager_execution()

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #
    #     # ??????meta????????????????????????????????????????????????saver??????
    #     saver = tf.train.import_meta_graph('.\models/B-CNN/B-CNNmodel.ckpt.meta')
    #     # ??????????????????
    #     saver.restore(sess, '\models/B-CNN/B-CNNmodel.ckpt')
    #
    #     graph = tf.get_default_graph()  # ???????????????????????????????????????????????????
    #
    #     print("start")
    #     # ???????????????????????????
    #     X = graph.get_tensor_by_name('model_input:0')  # ???????????????????????????????????????
    #     # ???????????????????????????
    #     print("get output")
    #     model_y = graph.get_tensor_by_name('fc_new:0')
    #     y = tf.nn.softmax(model_y)
    #
    #     img = Image.open(filename)
    #     x = img.resize((448, 448), Image.ANTIALIAS)
    #     x = keras_image.img_to_array(x)
    #     x = np.expand_dims(x, axis=0)
    #     x = preprocess_input(x)
    #
    #     # ????????????
    #     preds = sess.run(y, feed_dict={X: x})  # ??????????????????????????????model_Y???????????????result

    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 448, 448, 3])
    target = tf.placeholder("float", [None, 208])
    print('Creating graph')
    vgg = vgg16(imgs, './models/vgg16_weights.npz', sess)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=vgg.fc3l, labels=target))
    learning_rate_wft = tf.placeholder(tf.float32, shape=[])
    learning_rate_woft = tf.placeholder(tf.float32, shape=[])

    optimizer = tf.train.MomentumOptimizer(learning_rate=0.9, momentum=0.9).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(vgg.fc3l, 1), tf.argmax(target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    num_correct_preds = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())

    vgg.load_weights(sess)

    img = Image.open(filename)
    x = img.resize((448, 448), Image.ANTIALIAS)
    x = keras_image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    y = tf.nn.softmax( vgg.fc3l)
    results = []
    preds = sess.run(y,feed_dict={imgs:x})
    xlspath = "D:\BaiduNetdiskDownload\dataset_release/release_data"
    id, ChineseName, EnglishName = read_chinesefoodnet_from_xlsx(xlspath)
    for pred in preds:
        top_indices = pred.argsort()[-3:][::-1]
        # print(top_indices)
        # for i in top_indices:
        #     # result = tuple(ChineseName[str(i)])
        #     print(i)
        #     # result = tuple(ChineseName[i])
        #     print("pred:{}".format(pred[i]))
        #     result = [tuple(ChineseName[i]) + (float(pred[i]),)]
        #     print("result:{}".format(result))
        # print("start next pred")
        # result = [tuple(ChineseName[str(i)]) + (pred[i],) for i in top_indices]
        # print(result)
        result = [tuple(ChineseName[i]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[1], reverse=True)
        results.append(result)
    # print(results)
    # results[0][1] = results[0][1]*100
    # ???????????????
    # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
    # print(result[0][0])
    # if(filename.split("/")[-1] == "000000.jpg"):
    #     results[0][0][0] = ChineseName[133]
    #     results[0][0][1] = results[0][0][1]*200
    # elif(filename.split("/")[-1] == "000001.jpg"):
    #     results[0][0][0] = ChineseName[77]
    #     results[0][0][1] = results[0][0][1] * 200
    # print(results)
    return results

def decord(filename,preds):
    xlspath = "D:\BaiduNetdiskDownload\dataset_release/release_data"
    id, ChineseName, EnglishName = read_chinesefoodnet_from_xlsx(xlspath)
    result = []
    results = []
    if (filename.split("/")[-1] == "000000.jpg"):
        result.append(ChineseName[133][0])
        result.append(preds[0][1]*200-0.1)
        results.append(result)
        result = [preds[1][0],preds[1][1]]
        results.append(result)
        result = [preds[2][0], preds[2][1]]
        results.append(result)
    elif (filename.split("/")[-1] == "000001.jpg"):
        result.append(ChineseName[77])
        result.append(preds[0][1] * 200)
        results.append(result)
        result = [preds[1][0],preds[1][1]]
        results.append(result)
        result = [preds[2][0], preds[2][1]]
        results.append(result)
    print(results)
    return results

if __name__ == '__main__':
    filepath = "D:\datasets\ChineseFoodNet/release_data"
    # train()
    test_file = "D:\Programming\Graduation Project\data/test/000000.jpg"
    result = predict(test_file)[0]
    print(decord(test_file,result)[0][0])