from parsers import COCOParser, YOLOParser
import os, argparse
from collections import Counter

def main(args):

    if args.parser == 'coco':
        parser = COCOParser(args.contrast)
    elif args.parser == 'yolo':
        parser = YOLOParser(args.contrast)
    else:
        raise Exception(f'Parser of type {args.parser} not supported')

    os.makedirs(args.dir, exist_ok=True)
    samples = parser.read_samples(args.masks)
    train_samples, val_samples = parser.train_val_split(samples)
    subsets = {'train': train_samples, 'val': val_samples}
    incremental_id = Counter({'img': 0, 'obj': 0})

    for name, split in subsets.items():
        parser.make_img_dir(args.dir, split, name)
        parser.make_annotations(split, name, incremental_id, args.dir)

    if args.parser == 'yolo':
        parser.make_names_file()
        parser.make_data_file()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--masks", default="trainset.dat", help="Path of file that lists all mask file paths")
    parser.add_argument("-d", "--dir", default="data/images", help="Destination directory for converted data")
    parser.add_argument("-c", "--contrast", default=0.15, help="Contrast value for conversion to PNG")
    parser.add_argument("-p", "--parser", default='coco', help="Type of parser")

    args = parser.parse_args()
    main(args)