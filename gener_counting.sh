

for file in $(ls /output/result);
do
    python gener_wrong.py --meta_file $file --root_dir /output/result --output_dir /output/wrong
done
