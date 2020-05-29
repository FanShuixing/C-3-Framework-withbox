
# val
for type in val_noslope2 val_slope2;
do
    python pascalvoc.py  --gtfolder /input0 --detfolder /output/result/$type --threshold 0.7
done

# train
# for type in train_meta;
# do
#     python pascalvoc.py  --gtfolder /input0 --detfolder /output/result/result_14_$type --threshold 0.7
# done