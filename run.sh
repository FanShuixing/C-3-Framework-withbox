
#训练
python train.py $1 $2 $3 $4

#找到model_path路径
for file in `ls /output/tf_dir`;
do
    echo $file 
done
echo $file
for each in $(ls /output/tf_dir/${file}/all*.pth);
do 
    echo $each 
done

model_path=${each}

#predict
root_dir='/input0/test_dataset'
for type in val_mask;
do
    echo ${model_path} 
    python gangjin_predict.py --root_dir $root_dir --output_dir /output/result/${type} --meta_name ${type} --model_path $model_path

done
