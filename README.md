### Prepare Data
原始数据为目标检测数据，将目标检测数据转换为人群密度计数数据  
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
```
#首先将数据转换为mat格式
python detection_to_mat.py
```
~~然后将数据通过matlab转换为模型需要的csv格式的数据,在matlab中运行mat_to_csv.py,需要用到datasets/get_density_map_gaussian.m。mat_to_csv.m中需要修改一下变量，Line3~Line6。~~用python mat_to_npy.py将mat数据转换成npy格式，以供模型处理。在实际运行中，有些数据是有问题的，生成的csv维度不等于想要的，所以筛除掉这些不符合条件的，将符合条件的写到train.txt/val.txt中 ，所以最终生成的数据格式为
```
|---images
    |--xxx.jpg
    |--xxx.jpg
|--train_label
    |--xxx.npy
|--val_label
    |--xxx.npy
|--train.txt
|--val.txt
```
### Train
```
python train.py
```

### Test
```
python test.py
```
