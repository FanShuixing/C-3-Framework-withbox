
root_dir='/input0'
model_path=''
#验证集
for type in val_noslope2 val_slope2;
do
    echo ${type}
    python predict_withbox.py --root_dir $root_dir --output_dir /output/result/${type} --meta_name ${type} --model_path $model_path
done

# 训练集
# for type in train_meta2;
# do
#     echo ${type}
#     python predict_withbox.py --root_dir $root_dir --output_dir /output/result/${type} --meta_name ${type} --model_path $model_path

# done