
root_dir='/input1/normal'
model_path='/output/tf_dir/04-03_10-19_SHHB_Res101_1e-05/all_ep_61_mae_0.9_mse_1.8.pth'
#v16版本的验证集
for type in normal slope occlude;
do
    echo ${type}
    python predict_withbox.py --root_dir $root_dir --output_dir /output/result/result_16_${type} --meta_name ${type}_v16 --model_path $model_path
done

# v14版本的验证集
root_dir='/input0/normal'
for type in normal slope occlude;
do
    echo ${type}
    python predict_withbox.py --root_dir $root_dir --output_dir /output/result/result_14_${type} --meta_name ${type}_v14 --model_path $model_path

done