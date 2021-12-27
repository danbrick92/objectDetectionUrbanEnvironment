import argparse
import glob
import os
import random

import numpy as np

from utils import get_module_logger

# Globals
SPLITS = {
    'train' : .8,
    'val' : .5
}


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    # Determine number of files from source
    source_dir = 'training_and_validation'
    files = np.array(os.listdir(data_dir + "/" + source_dir))
    # Shuffle files
    np.random.shuffle(files)
    # Split up data
    end_train = int(files.shape[0] * SPLITS['train'])
    train = files[0:end_train]
    test_val = files[end_train:]
    end_val = int(test_val.shape[0] * SPLITS['val'])
    val = test_val[0:end_val]
    test = test_val[end_val:]
    # Move
    for item in train:
        source = data_dir + "/" + source_dir + "/" + item
        dest = data_dir + "/" + 'train/' + item
        os.replace(source, dest)
    for item in val:
        source = data_dir + "/" + source_dir + "/" + item
        dest = data_dir + "/" + 'val/' + item
        os.replace(source, dest)
    for item in test:
        source = data_dir + "/" + source_dir + "/" + item
        dest = data_dir + "/" + 'test/' + item
        os.replace(source, dest)
        

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', default='/home/workspace/data/waymo',
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)