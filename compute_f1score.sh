
# v16
for type in normal;
do
    python pascalvoc.py  --gtfolder /input0 --detfolder /output/result/result_16_$type --threshold 0.7
done

# v14
for type in normal;
do
    python pascalvoc.py  --gtfolder /input0 --detfolder /output/result/result_14_$type --threshold 0.7
done