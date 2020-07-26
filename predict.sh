
root_dir='/input0'
model_path='/output/tf_dir/07-22_01-36/all_ep_63_mae_6.7_mse_7.8.pth'
#验证集
# for type in val_mask;
# do
#     echo ${type}
#     python predict_withbox.py --root_dir $root_dir --output_dir /output/result/${type} --meta_name ${type} --model_path $model_path
# done

#测试集
root_dir='/input0/test_dataset'
for type in val_mask;
do
    echo ${type}
    python gangjin_predict.py --root_dir $root_dir --output_dir /output/result/${type} --meta_name ${type} --model_path $model_path

done