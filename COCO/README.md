# COCO Data Parser

Converts FITS mask data in [COCO format](https://cocodataset.org/#format-data) and creates train and val splits

## Args
- `-m`, Path of file that lists all mask file paths (trainset.dat)
- `-d` Destination directory for converted data (default: coco)

## Usage

Extract the whole content of the MLDataset_cleaned.tar.gz archive in the same directory as the parser.py script and launch it.
The result will be stored in the coco directory (or in another one if specified)
