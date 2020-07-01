#训练
#./run.sh --optimizer=adam --wh_decay=0.01 --offset_decay=0.1 --pos_decay=0.1
# python train.py $1 $2 $3 $4

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
root_dir='/input0/'
for type in  val_meta1265 val_noslope1265 val_slope1265;
do
    echo ${type}
    python predict_withbox.py --root_dir $root_dir --output_dir /output/result/${type} --meta_name ${type} --model_path $model_path
done

#generate counting error
for file in $(ls /output/result);
do
    python gener_wrong.py --meta_file $file --root_dir /output/result --output_dir /output/wrong
done

#compute f1-score
for type in val_meta1265 val_noslope1265 val_slope1265;
do
    python pascalvoc.py  --gtfolder /input0 --detfolder /output/result/$type --threshold 0.7
done