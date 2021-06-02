# data_parsers
Collection of data parsers in different formats for training neural network models on radioastronomical datasets

## COCO Parser
Converts FITS mask data in [COCO format](https://cocodataset.org/#format-data) and creates train and val splits

## YOLO Parser
Converts FITS mask data in [YOLO format](https://github.com/cvjena/darknet/blob/master/README.md) and creates train and val splits

### Args
- `-p` Type of parser (default: coco)
- `-m`, Path of file that lists all mask file paths (trainset.dat)
- `-d` Destination directory for converted data (default: coco)
- `-c` Contrast value for conversion from FITS to PNG (default: 0.15)

### Usage

Extract the whole content of the MLDataset_cleaned.tar.gz archive, then copy the `main.py` and `parsers.py` script and launch `python main.py`.
The result will be stored in a directory named `coco` (or in another one if specified)