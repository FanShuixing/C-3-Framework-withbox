
root_dir='/input0/test_dataset'
model_path=''
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