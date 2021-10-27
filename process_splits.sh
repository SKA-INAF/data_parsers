for split in train val test; do
    echo $split
    python main.py --data_dir data_splitter/splits --out_dir radio-galaxy --masks mask_list.dat --split $split
done