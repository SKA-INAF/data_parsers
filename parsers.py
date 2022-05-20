from genericpath import exists
import random, json, os, math
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import cv2
from pathlib import Path
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

    def __init__(self, dst_dir, split):
        self.CLASSES = {'sidelobe': 1, 'source': 2, 'galaxy': 3}
        self.dst_dir = dst_dir
        self.split = split
        self.error_file = Path(f'{self.split}_skipped.txt')
        if self.error_file.exists():
            self.error_file.unlink()

    def read_samples(self, trainset_path):
        '''trainset.dat file parsing to get dataset samples'''
        samples = []
        with open(trainset_path) as f:
            for json_path in tqdm(f):
                # json_path = json_path.replace('/home/riggi/Data/MLData', os.path.abspath(os.pardir))
                # json_path = os.path.normpath(json_path).strip()
                json_rel_path = Path(json_path.strip())
                abs_json_path = trainset_path.parent / json_rel_path
                with open(abs_json_path, 'r') as label_json:
                    label = json.load(label_json)
                    # replacing relative path with the absolute one
                    label['img'] = trainset_path.parent / Path('imgs') / label['img']
                    # label['img'] = label['img'].replace('..', os.sep.join(json_path.split(os.sep)[:-2]))
                    # label['img'] = os.path.normpath(label['img'])
                    samples.append(label)

            return samples

    def train_val_split(self, samples, val_ratio=0.1, test_ratio=0.005):
        '''trainset.dat file parsing to get a random train-val split'''

        
        random.shuffle(samples)
        test_sep = math.floor(len(samples) * test_ratio)
        val_sep = math.floor(len(samples) * val_ratio)
        assert (len(samples) / test_sep + val_sep) > 0.8, f'Less than 80% of data for training'
        val_entries = samples[ : val_sep]
        test_entries = samples[val_sep : val_sep + test_sep ]
        train_entries = samples[val_sep + test_sep : ]

        return train_entries, val_entries, test_entries

    @abstractmethod
    def make_img_dir(entries, split):
        '''Copies images into train or val folder'''
        return

    def log_error(self, msg):
        with open(self.error_file, 'a') as td:
            td.write(msg + '\n')


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

        if not contours:
            return None, None

        for xy in contours[0]:
            x_points.append(xy[0][0])
            y_points.append(xy[0][1])
        return x_points, y_points
    
    def get_img_size(self, img_path):
        '''Extracts size from the mask image'''
        img = fits.getdata(img_path).astype(np.uint8)
        return img.shape

    @abstractmethod
    def make_annotations(self, samples, split, incremental_id):
        '''Creates the JSON COCO annotations to be stored'''
        return


class COCOParser(DefaultParser):

    def __init__(self, contrast, dst_dir, split):
        super(COCOParser, self).__init__(dst_dir, split)
        self.contrast = contrast

    def make_img_dir(self, dst_dir, entries, split):
        '''Copies images into train or val folder'''
        split_path = dst_dir / Path(split)
        split_path.mkdir(exist_ok=True)
        dst_folder = split_path #/  Path('imgs')
        for line in tqdm(entries):
            img_name = line['img'].name # take image name
            img_name = img_name.replace('.fits', '.png')
            dst_folder.mkdir(exist_ok=True)
            dst_path = dst_folder / img_name
            line['filename'] = img_name
            self.fits_to_png(line['img'], dst_path, contrast=self.contrast)

    def make_annotations(self, samples, split, incremental_id, dst_dir):
        '''Creates the JSON COCO annotations to be stored'''

        coco_samples = { 'images':[], 'annotations':[], 'categories': [
                                                            {"id":1, "name": 'sidelobe'},
                                                            {"id":2, "name":'source'},
                                                            {"id":3, "name":'galaxy'},
                                                                ] 
                            }

        for line in tqdm(samples):

            h, w = fits.getdata(line['img']).shape
            image = {'id': incremental_id['img'], 'width': w, 'height': h, 'file_name': line['img'].name.replace('.fits', '.png')} # file_name = sample1_galaxy0001.png
            coco_samples['images'].append(image)

            for obj in line['objs']:
                if obj['class'] == '':
                    # probably for misannotation, the class is missing in some samples, which will be skipped 
                    self.log_error(f'Object {obj["mask"]} has no class')
                    continue
                # replaces the last two steps of the path with the steps to reach the mask file
                # mask_path = os.path.join(os.sep.join(line['img'].split(os.sep)[:-2]), 'masks', obj['mask'])
                mask_path = line['img'].parent.parent / Path('masks') / obj['mask']
                x_points, y_points = self.get_mask_coords(mask_path)

                if not (x_points and y_points):
                    self.log_error(f'Mask {mask_path} is empty')
                    continue

                poly = [(x, y) for x, y in zip(x_points, y_points)]
                # Flatten the array
                poly = [p for x in poly for p in x]

                if len(poly) <= 4:
                    # Eliminates annotations with segmentation masks with only 2 coordinates,
                    # which bugs the coco API
                    id = image['id']
                    filename = image['file_name']
                    msg = f'Invalid mask for file: {filename}\tlen: {len(poly)} (should be > 4)\t objs: {len(line["objs"])}'
                    self.log_error(msg)
                    continue

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

        return coco_samples


    def dump_annotations(self, dst_dir, split, coco_samples):
        annot_dst_dir = dst_dir / Path('annotations')
        annot_dst_dir.mkdir(exist_ok=True)
        annot_dst_path = annot_dst_dir / Path(f'{split}.json')
        with open(annot_dst_path, 'w') as out:
            print(f'Dumping data in file {split}.json')
            json.dump(coco_samples, out, indent=2, cls=NpEncoder)






class YOLOParser(DefaultParser):

    def __init__(self, contrast, dst_dir, split):
        super(YOLOParser, self).__init__(dst_dir, split)
        self.contrast = contrast

    def make_img_dir(self, dst_dir, entries, split):
        '''Copies images into train or val folder'''
        image_dir = dst_dir / Path('images')
        image_dir.mkdir(exist_ok=True)
        image_split_dir = image_dir / Path(split)
        image_split_dir.mkdir(exist_ok=True)

        with open(f'{split}.txt', 'w') as txt:
            for line in tqdm(entries):
                img_name = line['img'].stem
                # img_name = img_name.replace('.fits', '.png')
                # sample = line['img'].split(os.sep)[-3] # take sample name
                dst_path = (image_split_dir / Path(img_name)).with_suffix('.png')
                # dst_path = os.path.join(image_dir, f"{sample}_{img_name}")
                # line['filename'] = f'{sample}_{img_name}'
                line['filename'] = img_name
                txt.write(str(dst_path) + '\n')
                self.fits_to_png(line['img'], dst_path, contrast=self.contrast)

    def make_annotations(self, dst_dir, samples, split, incremental_id):
        '''Creates the text file annotations to be stored'''

        dst_dir = dst_dir / Path('labels')
        dst_dir.mkdir(exist_ok=True)
        dst_dir = dst_dir / Path(split)
        dst_dir.mkdir(exist_ok=True)

        for line in tqdm(samples):

            dst_path = (dst_dir / Path(line['filename']).with_suffix('.txt'))

            with open(dst_path, 'w') as obj_file:

                for obj in line['objs']:
                    if obj['class'] == '':
                        # probably for misannotation, the class is missing in some samples, which will be skipped 
                        continue
                    # replaces the last two steps of the path with the steps to reach the mask file
                    mask_path = os.path.join(os.sep.join(str(line['img']).split(os.sep)[:-2]), 'masks', obj['mask'])
                    x_points, y_points = self.get_mask_coords(mask_path)

                    if not (x_points and y_points):
                        self.log_error(f'Mask {mask_path} is empty')
                        continue

                    w, h = self.get_img_size(mask_path)

                    x_center = (np.max(x_points) + np.min(x_points)) / 2
                    y_center = (np.max(y_points) + np.min(y_points)) / 2
                    box_width = np.max(x_points) - np.min(x_points)
                    box_height = np.max(y_points) - np.min(y_points)

                    # Normalize coordinates
                    x_center = x_center / w
                    y_center = y_center / h
                    box_width = box_width / w 
                    box_height = box_height / h


                    if x_center < 0 or y_center < 0 or \
                        box_width < 0 or box_height < 0:
                        self.log_error(f'Box format for {mask_path} is invalid')
                        continue

                    obj_file.write(f'{self.CLASSES[obj["class"]]} {x_center} {y_center} {box_width} {box_height}\n')

                    incremental_id.update({'obj': 1})

                incremental_id.update({'img': 1})

    def make_data_file(self, dst_dir):
        data_file = dst_dir / Path('radiogalaxy.yaml')
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