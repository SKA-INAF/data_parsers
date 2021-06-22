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

    output_dir = os.path.join(os.path.abspath(os.pardir), args.data_dir, args.out_dir)
    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f'Reading samples from JSON {args.masks}...')
    mask_file = os.path.join(os.path.abspath(os.pardir), args.data_dir, args.masks)
    mask_file = os.path.normpath(mask_file)
    samples = parser.read_samples(mask_file)

    train_samples, val_samples = parser.train_val_split(samples)
    subsets = {'train': train_samples, 'val': val_samples}
    incremental_id = Counter({'img': 0, 'obj': 0})

    for split, samples in subsets.items():
        print(f'Making {split} image directory')
        parser.make_img_dir(output_dir, samples, split)
        print(f'Making {split} annotation directory')
        parser.make_annotations(samples, split, incremental_id, output_dir)

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

    args = parser.parse_args()
    main(args)