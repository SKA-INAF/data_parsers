import random, json, os, math
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import cv2
from torchvision.utils import save_image
from abc import ABC, abstractmethod

from astropy.visualization import ZScaleInterval

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

class DefaultParser(ABC):

    def __init__(self):
        self.CLASSES = {'galaxy': 1, 'source': 2, 'sidelobe': 3}

    def read_samples(self, trainset_path):
        '''trainset.dat file parsing to get dataset samples'''
        samples = []
        with open(trainset_path) as f:
            for json_path in tqdm(f):
                json_path = json_path.replace('/home/riggi/Data/MLData', os.path.abspath(os.pardir))
                json_path = os.path.normpath(json_path).strip()
                with open(json_path, 'r') as label_json:
                    label = json.load(label_json)
                    # replacing relative path with the absolute one
                    label['img'] = label['img'].replace('..', os.sep.join(json_path.split(os.sep)[:-2]))
                    label['img'] = os.path.normpath(label['img'])
                    samples.append(label)

            return samples

    def train_val_split(self, samples, train_ratio=0.8):
        '''trainset.dat file parsing to get a random train-val split'''

        random.shuffle(samples)
        sep = math.floor(len(samples) * train_ratio)
        train_entries = samples[:sep]
        val_entries = samples[sep:]

        return train_entries, val_entries

    @abstractmethod
    def make_img_dir(entries, split):
        '''Copies images into train or val folder'''
        return

    def fits_to_png(self, file_path, dst_path, contrast=0.15):
        
        img = fits.getdata(file_path, ignore_missing_end=True)
        interval = ZScaleInterval(contrast=contrast)
        min, max = interval.get_limits(img)

        img = (img-min)/(max-min)

        save_image(torch.from_numpy(img), dst_path)

    def get_mask_coords(self, mask_path):
        '''Extracts coordinates from the mask image'''
        img = fits.getdata(mask_path).astype(np.uint8)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        x_points = []
        y_points = []

        for xy in contours[0]:
            x_points.append(xy[0][0])
            y_points.append(xy[0][1])
        return x_points, y_points

    @abstractmethod
    def make_annotations(self, samples, split, incremental_id):
        '''Creates the JSON COCO annotations to be stored'''
        return


class COCOParser(DefaultParser):

    def __init__(self, contrast):
        super(COCOParser, self).__init__()
        self.contrast = contrast

    def make_img_dir(self, dst_dir, entries, split):
        '''Copies images into train or val folder'''
        os.makedirs(os.path.join(dst_dir, split), exist_ok=True)
        for line in tqdm(entries):
            img_name = line['img'].split(os.sep)[-1] # take image name
            img_name = img_name.replace('.fits', '.png')
            sample = line['img'].split(os.sep)[-3] # take sample name
            dst_path = os.path.join(dst_dir, split, f"{sample}_{img_name}")
            line['filename'] = f'{sample}_{img_name}'
            self.fits_to_png(line['img'], dst_path, contrast=self.contrast)


    def make_annotations(self, samples, split, incremental_id, dst_dir):
        '''Creates the JSON COCO annotations to be stored'''

        coco_samples = { 'images':[], 'annotations':[], 'categories': [
                                                            {"id":1, "name": 'galaxy'},
                                                            {"id":2, "name":'source'},
                                                            {"id":3, "name":'sidelobe'},
                                                                ] 
                            }

        for line in tqdm(samples):

            w, h = fits.getdata(line['img']).shape
            image = {'id': incremental_id['img'], 'width': w, 'height': h, 'file_name': line['filename']}
            coco_samples['images'].append(image)

            for obj in line['objs']:
                if obj['class'] == '':
                    # probably for misannotation, the class is missing in some samples, which will be skipped 
                    continue
                # replaces the last two steps of the path with the steps to reach the mask file
                mask_path = os.path.join(os.sep.join(line['img'].split(os.sep)[:-2]), 'masks', obj['mask'])
                x_points, y_points = self.get_mask_coords(mask_path)

                poly = [(x, y) for x, y in zip(x_points, y_points)]
                # Flatten the array
                poly = [p for x in poly for p in x]

                if len(poly) < 4:
                    # Eliminates annotations with segmentation masks with only 2 coordinates,
                    # which bugs the coco API
                    continue
                    # with open(f'{split}_to_del.txt', 'a') as td:
                    #     id = image['id']
                    #     filename = image['file_name']
                    #     td.write(f'ID: {id}\tfile: {filename}\tlen: {len(poly)}\n objs: {len(line["objs"])}')

                x0, y0, x1, y1 = np.min(x_points), np.min(y_points), np.max(x_points), np.max(y_points) 
                w, h = x1 - x0, y1 - y0 
                area = w * h
                
                annotation = {
                    'id': incremental_id['obj'], 
                    'category_id': self.CLASSES[obj['class']],
                    'image_id': incremental_id['img'], 
                    'segmentation': [poly],
                    'area': area,
                    "bbox": [x0, y0, w, h],
                    'iscrowd': 0
                }

                coco_samples['annotations'].append(annotation)

                incremental_id.update({'obj': 1})

            incremental_id.update({'img': 1})


        os.makedirs(os.path.join(dst_dir, 'annotations'), exist_ok=True)
        with open(os.path.join(dst_dir, 'annotations', f'{split}.json'), 'w') as out:
            print(f'Dumping data in file {split}.json')
            json.dump(coco_samples, out, indent=2, cls=NpEncoder)


class YOLOParser(DefaultParser):

    def __init__(self, contrast):
        super(YOLOParser, self).__init__()
        self.contrast = contrast

    def make_img_dir(self, dst_dir, entries, split):
        '''Copies images into train or val folder'''
        image_dir = os.path.join(dst_dir, 'images', split)
        os.makedirs(image_dir, exist_ok=True)

        with open(f'{split}.txt', 'w') as txt:
            for line in tqdm(entries):
                img_name = line['img'].split(os.sep)[-1] # take image name
                img_name = img_name.replace('.fits', '.png')
                sample = line['img'].split(os.sep)[-3] # take sample name
                dst_path = os.path.join(image_dir, f"{sample}_{img_name}")
                line['filename'] = f'{sample}_{img_name}'
                txt.write(dst_path + '\n')
                self.fits_to_png(line['img'], dst_path, contrast=self.contrast)

    def make_annotations(self, dst_dir, samples, split, incremental_id):
        '''Creates the JSON COCO annotations to be stored'''

        dst_dir = os.path.join(dst_dir, 'labels', split)
        os.makedirs(dst_dir, exist_ok=True)

        for line in tqdm(samples):

            dst_path = os.path.join(dst_dir, line['filename'].replace('.png', '.txt'))

            with open(dst_path, 'w') as obj_file:

                for obj in line['objs']:
                    if obj['class'] == '':
                        # probably for misannotation, the class is missing in some samples, which will be skipped 
                        continue
                    # replaces the last two steps of the path with the steps to reach the mask file
                    mask_path = os.path.join(os.sep.join(line['img'].split(os.sep)[:-2]), 'masks', obj['mask'])
                    x_points, y_points = self.get_mask_coords(mask_path)

                    x_center = (np.max(x_points) + np.min(x_points)) / 2
                    y_center = (np.max(y_points) + np.min(y_points)) / 2
                    box_width = np.max(x_points) - np.min(x_points)
                    box_height = np.max(y_points) - np.min(y_points)

                    obj_file.write(f'{self.CLASSES[obj["class"]]} {x_center} {y_center} {box_width} {box_height}')

                    incremental_id.update({'obj': 1})

                incremental_id.update({'img': 1})

    def make_data_file(self, dst_dir):
        data_file = os.path.join(dst_dir, 'radiogalaxy.yaml')
        with open(data_file, 'w') as out:
            out.write('# Number of classes')
            out.write(f'\nnc: {len(self.CLASSES)}\n')
            out.write('\n# Train and val directories')
            out.write(f'\ntrain: data/images/train/')
            out.write(f'\nval: data/images/val/')
            out.write(f'\nnames: [ ')
            for name in self.CLASSES:
                out.write(f'\'{name}\', ')
            out.write(f']')