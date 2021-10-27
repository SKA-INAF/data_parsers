from genericpath import exists
from parsers import COCOParser, YOLOParser
import os, argparse
from pathlib import Path
from collections import Counter

def main(args):

    parent_dir = Path().resolve().parent
    data_dir = parent_dir / Path(args.data_dir) / Path(args.split)
    output_dir = Path(args.out_dir)
    split_dir = Path(args.split)

    output_dir.mkdir(exist_ok=True)

    if args.parser == 'coco':
        parser = COCOParser(args.contrast, output_dir, split_dir)
    elif args.parser == 'yolo':
        parser = YOLOParser(args.contrast, output_dir, split_dir)
    else:
        raise Exception(f'Parser of type {args.parser} not supported')

    

    print(f'Reading samples from JSON {args.masks}...')
    mask_file = data_dir / Path(args.masks)
    samples = parser.read_samples(mask_file)

    # train_samples, val_samples, test_samples = parser.train_val_split(samples)
    # subsets = {
    #     'train': train_samples,
    #     'val': val_samples,
    #     'test': test_samples
    # }
    incremental_id = Counter({'img': 0, 'obj': 0})

    print(f'Making {args.split} image directory')
    # parser.make_img_dir(output_dir, samples, args.split)
    print(f'Making {args.split} annotation directory')
    coco_samples = parser.make_annotations(samples, args.split, incremental_id, output_dir)
    parser.dump_annotations(output_dir, split_dir, coco_samples)

    if args.parser == 'yolo':
        print('Creating data file...')
        parser.make_data_file(output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data_dir", default="MLDataset_cleaned", help="Path of whole dataset directory (should be a sibling directory of the parsers one)")
    parser.add_argument("-m", "--masks", default="trainset.dat", help="Path of file that lists all mask file paths")
    parser.add_argument("-o", "--out_dir", default="data/", help="Destination directory for converted data")
    parser.add_argument("-c", "--contrast", default=0.15, help="Contrast value for conversion to PNG")
    parser.add_argument("-p", "--parser", default='coco', help="Type of parser")
    parser.add_argument("-s", "--split", choices=['train', 'val', 'test'], help="Split to parse")

    args = parser.parse_args()
    main(args)