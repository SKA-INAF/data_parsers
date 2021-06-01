import random, json, os, math
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import cv2
from torchvision.utils import save_image

from astropy.visualization import ZScaleInterval
from collections import Counter

import argparse

CLASSES = {'galaxy': 1, 'source': 2, 'sidelobe': 3}

class NpEncoder(json.JSONEncoder):
    # JSON Encoder class to manage output file saving
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def train_val_split(trainset_path, train_ratio=0.8):
    '''trainset.dat file parsing to get a random train-val split'''

    samples = []
    with open(trainset_path) as f:
        for json_path in tqdm(f):
            json_path = json_path.replace('/home/riggi/Data/MLData/MLDataset_cleaned', os.path.join(os.path.abspath(os.getcwd())))
            json_path = os.path.normpath(json_path).strip()
            with open(json_path, 'r') as label_json:
                label = json.load(label_json)
                # replacing relative path with the absolute one
                label['img'] = label['img'].replace('..', '\\'.join(json_path.split('\\')[:-2]))
                label['img'] = os.path.normpath(label['img'])
                samples.append(label)

        random.shuffle(samples)
        sep = math.floor(len(samples) * train_ratio)
        train_entries = samples[:sep]
        val_entries = samples[sep:]

        return train_entries, val_entries

def make_img_dir(entries, split):
    '''Copies images into train or val folder'''
    os.makedirs(os.path.join(args.dir, split), exist_ok=True)
    for line in entries:
        img_name = line['img'].split('\\')[-1] # take image name
        img_name = img_name.replace('.fits', '.png')
        sample = line['img'].split('\\')[-3] # take sample name
        dst_path = os.path.join(args.dir, split, f"{sample}_{img_name}")
        line['filename'] = f'{sample}_{img_name}'
        fits_to_png(line['img'], dst_path, contrast=args.contrast)

def fits_to_png(file_path, dst_path, contrast=0.15):
    
    img = fits.getdata(file_path, ignore_missing_end=True)
    interval = ZScaleInterval(contrast=contrast)
    min, max = interval.get_limits(img)

    img = (img-min)/(max-min)

    save_image(torch.from_numpy(img), dst_path)

def get_mask_coords(mask_path):
    '''Extracts coordinates from the mask image'''
    img = fits.getdata(mask_path).astype(np.uint8)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    x_points = []
    y_points = []

    for xy in contours[0]:
        x_points.append(xy[0][0])
        y_points.append(xy[0][1])
    return x_points, y_points


def make_annotations(samples, split, incremental_id):
    '''Creates the JSON COCO annotations to be stored'''

    coco_samples = { 'images':[], 'annotations':[], 'categories': [
                                                        {"id":1, "name": 'galaxy'},
                                                        {"id":2, "name":'source'},
                                                        {"id":3, "name":'sidelobe'},
                                                            ] 
                        }

    for line in tqdm(samples):

        w, h = fits.getdata(line['img']).shape
        image = {'id': incremental_id['img'], 'width': w, 'height': h, 'file_name': line['filename'], 'annotations': [] }
        coco_samples['images'].append(image)

        for obj in line['objs']:
            if obj['class'] == '':
                # probably for misannotation, the class is missing in some samples, which will be skipped 
                continue
            # replaces the last two steps of the path with the steps to reach the mask file
            mask_path = os.path.join('\\'.join(line['img'].split('\\')[:-2]), 'masks', obj['mask'])
            x_points, y_points = get_mask_coords(mask_path)


            poly = [(x, y) for x, y in zip(x_points, y_points)]
            # Flatten the array
            poly = [p for x in poly for p in x]

            area = (np.max(x_points) - np.min(x_points)) * (np.max(y_points) - np.min(y_points))
            
            annotation = {
                'id': incremental_id['obj'], 
                'category_id': CLASSES[obj['class']],
                'image_id': incremental_id['img'], 
                'segmentation': [poly],
                'area': area, 
                "bbox": [np.min(x_points), np.min(y_points), np.max(x_points), np.max(y_points)],
                'iscrowd': 0
            }

            # coco_samples['annotations'].append(annotation)
            image['annotations'].append(annotation)

            incremental_id.update({'obj': 1})

        incremental_id.update({'img': 1})
        
    os.makedirs(os.path.join(args.dir, 'annotations'), exist_ok=True)
    with open(os.path.join(args.dir, 'annotations', f'{split}.json'), 'w') as out:
        print(f'Dumping data in file {split}.json')
        json.dump(coco_samples, out, indent=2, cls=NpEncoder)

def main(args):
    os.makedirs(args.dir, exist_ok=True)
    train_samples, val_samples = train_val_split(args.masks)
    subsets = {'train': train_samples, 'val': val_samples}
    incremental_id = Counter({'img': 0, 'obj': 0})
    for name, split in subsets.items():
        make_img_dir(split, name)
        make_annotations(split, name, incremental_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # File listing all json files that contain mask information
    parser.add_argument("-m", "--masks", default="trainset.dat", help="Path of file that lists all mask file paths")

    # Directory where all output files will be saved
    parser.add_argument("-d", "--dir", default="radio-galaxy", help="Destination directory for converted data")

    # Contrast value in conversion to PNG
    parser.add_argument("-c", "--contrast", default=0.15, help="Contrast value for conversion to PNG")

    args = parser.parse_args()
    main(args)
