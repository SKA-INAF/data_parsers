# data_parsers
Collection of data parsers in different formats for training neural network models on radioastronomical datasets
Parses a single split of the dataset, so the split has to be preventively done when running this script

## COCO Parser
Converts FITS mask data in [COCO format](https://cocodataset.org/#format-data)

## YOLO Parser
Converts FITS mask data in [YOLO format](https://github.com/cvjena/darknet/blob/master/README.md)

### Args
- `-p` Type of parser (default: coco)
- `-m`, Path of file that lists all mask file paths (trainset.dat)
- `-d` Destination directory for converted data (default: coco)
- `-c` Contrast value for conversion from FITS to PNG (default: 0.15)

### Usage

Extract the whole content of the `MLDataset_cleaned.tar.gz` archive, perform the split using the [data_splitter](https://github.com/SKA-INAF/data_splitter) script, then put the folders in the same parent directory (next section provides a visual hint) and launch `sh process_splits.sh` to run the script for all splits.
The result will be stored in a directory named `coco` (or in another one if specified)

### Directory Structure
```
parent_folder
└───data_parsers
│   │───main.py
│   │───...
│   │───README.md (**YOU ARE HERE**)    
│   │
└───data_dir (e.g. MLDataset_cleaned)
    │
    └───train
    │   │───imgs
    │   │───annotations
    │   │───masks
    │   │───imgs_png
    │   │───...
    └───val
    │   │───imgs
    │   │───annotations
    │   │───masks
    │   │───imgs_png
    │   │───...
    └───test
        │───imgs
        │───annotations
        │───masks
        │───imgs_png
        │───...

```