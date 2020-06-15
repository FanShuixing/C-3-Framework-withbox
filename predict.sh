
root_dir='/input0/test_dataset'
model_path='/output/tf_dir/06-15_07-24/all_ep_71_mae_2.3_mse_3.1.pth'
model_path='/output/tf_dir/06-15_11-41/all_ep_51_mae_3.0_mse_3.8.pth'
#验证集
# for type in val_noslope2 val_slope2;
# do
#     echo ${type}
#     python predict_withbox.py --root_dir $root_dir --output_dir /output/result/${type} --meta_name ${type} --model_path $model_path
# done

# 训练集
for type in val_mask;
do
    echo ${type}
    python gangjin_predict.py --root_dir $root_dir --output_dir /output/result/${type} --meta_name ${type} --model_path $model_path

done