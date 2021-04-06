#  Graduation-Project
该项目为我的毕设项目，主要功能为完成对菜品图片的识别工作

环境需求：TensorFlow2.0+

[TOC]

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