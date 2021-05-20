from matplotlib import pyplot as plt
import numpy as np
import math

def sigmoid(x):
    """sigmoid函数"""
    return 1 / (1 + np.exp(-x))

def dx_sigmoid(x):
    """sigmoid函数的导数"""
    return sigmoid(x) * (1 - sigmoid(x))

def tanh (x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def relu (x):
    return np.maximum(x, 0)

def draw_sigmoid():
    x = np.arange(-10, 10, 0.01)
    fx = sigmoid(x)
    dx_fx = dx_sigmoid(x)
    # ---------------------------------------------
    plt.figure(figsize=(8,8))
    plt.subplot(2,1,1)
    ax = plt.gca()  # 得到图像的Axes对象
    ax.spines['right'].set_color('none')  # 将图像右边的轴设为透明
    ax.spines['top'].set_color('none')  # 将图像上面的轴设为透明
    ax.xaxis.set_ticks_position('bottom')  # 将x轴刻度设在下面的坐标轴上
    ax.yaxis.set_ticks_position('left')  # 将y轴刻度设在左边的坐标轴上
    ax.spines['bottom'].set_position(('data', 0))  # 将两个坐标轴的位置设在数据点原点
    ax.spines['left'].set_position(('data', 0))
    plt.title('Sigmoid Function')
    plt.xlabel('x')
    plt.ylabel('fx')
    plt.plot(x, fx)

    plt.subplot(2, 1, 2)
    ax = plt.gca()  # 得到图像的Axes对象
    ax.spines['right'].set_color('none')  # 将图像右边的轴设为透明
    ax.spines['top'].set_color('none')  # 将图像上面的轴设为透明
    ax.xaxis.set_ticks_position('bottom')  # 将x轴刻度设在下面的坐标轴上
    ax.yaxis.set_ticks_position('left')  # 将y轴刻度设在左边的坐标轴上
    ax.spines['bottom'].set_position(('data', 0))  # 将两个坐标轴的位置设在数据点原点
    ax.spines['left'].set_position(('data', 0))
    plt.title('Derivative of Sigmoid Function')
    plt.xlabel('x')
    plt.ylabel('dx_fx')
    plt.plot(x, dx_fx)
    plt.show()

def draw_tanh():
    x = np.arange(-10, 10)
    y1 = tanh(x)
    plt.xlim(-11, 11)
    ax = plt.gca() # get current axis 获得坐标轴对象
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')# 将右边 上边的两条边颜色设置为空 其实就相当于抹掉这两条边
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left') # 指定下边的边作为 x 轴   指定左边的边为 y 轴
    ax.spines['bottom'].set_position(('data', 0))# 指定 data  设置的bottom(也就是指定的x轴)绑定到y轴的0这个点上
    ax.spines['left'].set_position(('data', 0))
    plt.plot(x, y1, label='Tanh', linestyle="-", color="green")  # label为标签
    plt.legend(['Tanh'])
    plt.show()
    # plt.savefig('Tanh.png', dpi=500)  # 指定分辨

def draw_relu():
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    x = np.arange(-10, 10)
    # y = np.where(x < 0, 0, x)  # 满足条件(condition)，输出x，不满足输出y
    y = relu(x)
    plt.xlim(-11, 11)
    plt.ylim(-11, 11)
    ax = plt.gca()# get current axis 获得坐标轴对象
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')# 将右边 上边的两条边颜色设置为空 其实就相当于抹掉这两条边
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')# 指定下边的边作为 x 轴   指定左边的边为 y 轴
    ax.spines['bottom'].set_position(('data', 0))# 指定 data  设置的bottom(也就是指定的x轴)绑定到y轴的0这个点上
    ax.spines['left'].set_position(('data', 0))

    plt.plot(x, y, label='ReLU', linestyle="-", color="darkviolet")  # label为标签
    plt.legend(['ReLU'])
    plt.show()

if __name__ == '__main__':
    draw_sigmoid()
    # draw_tanh()
    # draw_relu()