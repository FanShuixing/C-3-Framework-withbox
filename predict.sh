
root_dir='/input0'
model_path='/output/tf_dir/06-18_08-00/all_ep_51_mae_2.9_mse_3.7.pth'
model_path='/output/tf_dir/06-18_11-50/all_ep_96_mae_7.0_mse_8.0.pth'
model_path='/output/tf_dir/08-15_01-11/all_ep_63_mae_6.7_mse_7.7.pth'
#验证集
for type in val_mask;
do
    echo ${type}
    python predict_withbox.py --root_dir $root_dir --output_dir /output/result/${type} --meta_name ${type} --model_path $model_path
done

root_dir='/input0/test_dataset'
for type in val_mask;
do
    echo ${type}
    python gangjin_predict.py --root_dir $root_dir --output_dir /output/result/${type} --meta_name ${type} --model_path $model_path

done