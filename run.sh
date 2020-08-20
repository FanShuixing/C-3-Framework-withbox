#训练
#./run.sh --optimizer=adam --wh_decay=0.01 --offset_decay=0.1 --pos_decay=0.1
# python train.py $1 $2 $3 $4
python train.py

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

#predict 验证集
root_dir='/input0'
for type in val_mask;
do
    echo ${type}
    python predict_withbox.py --root_dir $root_dir --output_dir /output/result/${type} --meta_name ${type} --model_path $model_path
done



#compute f1-score
for type in val_mask;
do
    python pascalvoc.py  --gtfolder /input0 --detfolder /output/result/$type --threshold 0.7
done

#测试集
root_dir='/input0/test_dataset'
for type in val_mask;
do
    echo ${type}
    python gangjin_predict.py --root_dir $root_dir --output_dir /output/result_test/${type} --meta_name ${type} --model_path $model_path

done



