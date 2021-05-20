from DenseNet import densenet_logs_1000
from VGG16 import logs_500,logs
import matplotlib.pyplot as plt

def compare_CNN(index):
    vgg_train_acc = logs['accuracy']
    vgg_train_loss = logs['loss']
    vgg_val_acc = logs['val_accuracy']
    vgg_val_loss = logs['val_loss']

    densenet_train_acc = densenet_logs_1000['acc'][:index]
    densenet_val_acc = densenet_logs_1000['val_acc'][:round(index/2)]
    densenet_train_loss = densenet_logs_1000['loss'][:index]
    densenet_val_loss = densenet_logs_1000['val_loss'][:round(index/2)]

    if(index >100):
        for i in range(index-100):
            vgg_train_acc.append(logs_500['acc'][i])
            vgg_train_loss.append(logs_500['loss'][i])
            if(i < len(logs_500['val_acc'])):
                vgg_val_loss.append(logs_500['val_loss'][i])
                vgg_val_acc.append(logs_500['val_acc'][i])

    plt.figure(figsize=(8, 8))
    plt.subplot(2,1,1)
    plt.title("Validation Accuracy")
    plt.plot(vgg_val_acc,label='VGG Validation Accuracy',color = "red")
    plt.plot(densenet_val_acc,label= 'DenseNet Validation Accuracy',color='green')
    plt.legend(['VGG','DenseNet'])

    plt.subplot(2,1,2)
    plt.title("Validation Loss")
    plt.plot(vgg_val_loss, label='VGG Validation Loss', color="red")
    plt.plot(densenet_val_loss, label='DenseNet Validation Loss', color='green')
    plt.legend(['VGG', 'DenseNet'])
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.title("Training Accuracy")
    plt.plot(vgg_train_acc, label='VGG Training Accuracy', color="red")
    plt.plot(densenet_train_acc, label='DenseNet Training Accuracy', color='green')
    plt.legend(['VGG', 'DenseNet'])

    plt.subplot(2, 1, 2)
    plt.title("Training Loss")
    plt.plot(vgg_train_loss, label='VGG Training Loss', color="red")
    plt.plot(densenet_train_loss, label='DenseNet Training Loss', color='green')
    plt.legend(['VGG', 'DenseNet'])
    plt.show()

if __name__ == "__main__":
    compare_CNN(500)