## 项目简介：
此项目基于C-3-Framework的计数框架，在其之上修改网络结构以支持输出每个物体的框(原来的框架是计数的框架，只支持输出计数的数量)。下面会以简单的流程介绍整个项目。

> 1.训练过程
>> - 数据准备
>> - 训练
>> - 测试  

> 2.本项目创新点  
>> - 网络结构更新
>> - 添加后处理
---
### TODO
- [ ] 添加与其他模型的对比效果
- [ ] 继续改进网络结构和损失函数

### 数据准备
原始数据为目标检测格式的数据，放置形式如下
```
|--image
   |--xxx.jpg
   |---xxx.jpg
|--mask_labels
   |--xxx.json
   |--xxx.json
|--train_meta.csv
|--val_meta.csv
```
### Train
```
python train.py
```

### Test
```
./predict.sh
```


### 网络结构修改
![img](https://github.com/FanShuixing/C-3-Framework-withbox/blob/python3.x/demo_imgs/network.png)  
如上图所示，红色箭头的分支是新添加的网络结构。
### 后处理
现在网络的输出为两部分，第一个分支同计数框架一样，输出密度图，第二个分支输出框，通过密度图得到关键点的坐标，将关键点的坐标转换为索引，然后根据索引找到对应关键点的框。
目标：根据密度图找到关键点的坐标。  
![img](https://github.com/FanShuixing/C-3-Framework-withbox/blob/python3.x/demo_imgs/flir_20191208T200014_rgb_image_pred_37.0_gt_31.0_box_32.png)  
上图是将网络的密度图叠加在了原图之上，后处理需要做的事情是找到关键点(即木材中心点位置的坐标)。如果每一个木材中心点对应的都是这个密度图的最大值，那么问题就变得非常简单。只要找到密度图的最大值，然后找到最大值对应的所有坐标，那么就可以得到所有关键点的坐标。
但是，网络输出的密度图并没有这个特点，会出现木材中心点周围的像素值可能比其他木材中心点的像素值高的情况，所以如果假设我们取整张图像前500个最大值(去除重复的最大值后，只有32个点),结果如下图所示。
![img](https://github.com/FanShuixing/C-3-Framework-withbox/blob/python3.x/demo_imgs/densitymap2.png)   
可以看出来，虽然所有木材的关键点都被找到了，但是一个木材都通常对应了很多个关键点，而我们只需要一个最靠近中心的关键点，所以接下来要对这些点计算距离，当距离小于10的时候，会被筛除掉。筛除之后的结果如下图![img](https://github.com/FanShuixing/C-3-Framework-withbox/blob/python3.x/demo_imgs/densitymap_Deduplication.jpg)  
这样我们就可以根据网络输出的密度图得到关键点的坐标，然后将坐标转换为索引，就可以根据索引得到每一个木材中心点对应的框。


