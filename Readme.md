#  Graduation-Project
该项目为我的毕设项目，主要功能为完成对菜品图片的识别工作

环境需求：TensorFlow2.0+



## 1 数据集读取

目前已完成对ChineseFoodNet的存储与下载工作。

### 1.1 数据集本体结构

ChineseFoodNet的数据集结构如下：

```bash
/datasets
	/train
		/000
			-*.jpg
			-*.jpg
			...
		/001
			-*.jpg
			...
		...
	/test
		-*.jpg
		-*.jpg
		...
	/val
		/000
			-*.jpg
			-*.jpg
			...
		/001
			-*.jpg
			...
		...
	-test_list.txt
	-test_truth_list.txt
	-train_list.txt
	-val_list.txt
```

其中`train_list.txt`及`val_list.txt`内容格式相同，如下所示：

```
000/000000.jpg 0
000/000001.jpg 0
```

`test_truth_list.txt`文件格式如下：

```
000000.jpg 133
000001.jpg 77
```

### 1.2 数据集读取

对数据集的读取工作，使用`Datasets_loader.py`文件下`data_loader(data_path, type)`函数即可。`data_path`指定为数据集根目录即可。该函数会返回一个tf.data.Datasets的Dataset，该Dataset包含了数据集及对应labels，可直接进行训练。
当然，对于大批量的数据读取，也准备了`load_data_by_keras`方法。

## 2 模型训练

### 2.1 模型介绍

本项目中，首先使用了VGG16及DenseNet网络，网络代码分别保存在`VGG16.py`，`DenseNet.py`下，两网络都分别编写了训练函数及对应的预测函数，可直接调用，即可完成对应功能。
Note:预测函数需要预先训练并保存后才可以进行预测

## 3 可视化功能

本项目添加了TensorBoard可通过该方法查看训练中、训练结束的模型loss及acc的迭代过程。

其外，还添加了一个UI，方便用户体验可视化的菜品预测效果，请run `visualize.py`查看。