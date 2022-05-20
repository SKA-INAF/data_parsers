for split in train val test; do
    echo $split
    python main.py --data_dir radio-galaxy --out_dir parsed_yolo --masks mask_list.dat --split $split --parser $1
done