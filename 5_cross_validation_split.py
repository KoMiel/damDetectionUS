
### in this script, files are split into three (roughly) equally sized parts for cross validation


### import packages

import shutil
from os import listdir
from os.path import isfile, join
import random
import numpy as np



random.seed(200) # random seed for reproducibility



### main script

# find directories
directory_main = 'data/rivers/'
directories_destination = ['data/rivers/1/', 'data/rivers/2/', 'data/rivers/3/']

# get files
files = [f for f in listdir(directory_main) if isfile(join(directory_main, f))]
sel = [s for s in files if "tfrecords" in s]

#shuffle filenames
random.shuffle(sel)
n_files = len(sel)

# loop over partitions
for i in range(3):
    # select files
    sel_part = sel[np.int32(i*n_files/3):np.int32((i+1)*n_files/3)]
    # get source path
    sel_source = [directory_main + s for s in sel_part]
    # get destination part
    sel_destination = [directories_destination[i] + s for s in sel_part]
    # move files to destination
    for j in range(len(sel_part)):
        shutil.move(sel_source[j], sel_destination[j])

# repeat for all other subsets of data
directory_main = 'data/lakes/'
directories_destination = ['data/lakes/1/', 'data/lakes/2/', 'data/lakes/3/']

files = [f for f in listdir(directory_main) if isfile(join(directory_main, f))]
sel = [s for s in files if "tfrecords" in s]
random.shuffle(sel)
n_files = len(sel)
for i in range(3):
    sel_part = sel[np.int32(i*n_files/3):np.int32((i+1)*n_files/3)]
    sel_source = [directory_main + s for s in sel_part]
    sel_destination = [directories_destination[i] + s for s in sel_part]
    for j in range(len(sel_part)):
        shutil.move(sel_source[j], sel_destination[j])

directory_main = 'data/very_small/'
directories_destination = ['data/very_small/1/', 'data/very_small/2/', 'data/very_small/3/']

files = [f for f in listdir(directory_main) if isfile(join(directory_main, f))]
sel = [s for s in files if "tfrecords" in s]
random.shuffle(sel)
n_files = len(sel)
for i in range(3):
    sel_part = sel[np.int32(i*n_files/3):np.int32((i+1)*n_files/3)]
    sel_source = [directory_main + s for s in sel_part]
    sel_destination = [directories_destination[i] + s for s in sel_part]
    for j in range(len(sel_part)):
        shutil.move(sel_source[j], sel_destination[j])

directory_main = 'data/small/'
directories_destination = ['data/small/1/', 'data/small/2/', 'data/small/3/']

files = [f for f in listdir(directory_main) if isfile(join(directory_main, f))]
sel = [s for s in files if "tfrecords" in s]
random.shuffle(sel)
n_files = len(sel)
for i in range(3):
    sel_part = sel[np.int32(i*n_files/3):np.int32((i+1)*n_files/3)]
    sel_source = [directory_main + s for s in sel_part]
    sel_destination = [directories_destination[i] + s for s in sel_part]
    for j in range(len(sel_part)):
        shutil.move(sel_source[j], sel_destination[j])

directory_main = 'data/medium/'
directories_destination = ['data/medium/1/', 'data/medium/2/', 'data/medium/3/']

files = [f for f in listdir(directory_main) if isfile(join(directory_main, f))]
sel = [s for s in files if "tfrecords" in s]
random.shuffle(sel)
n_files = len(sel)
for i in range(3):
    sel_part = sel[np.int32(i*n_files/3):np.int32((i+1)*n_files/3)]
    sel_source = [directory_main + s for s in sel_part]
    sel_destination = [directories_destination[i] + s for s in sel_part]
    for j in range(len(sel_part)):
        shutil.move(sel_source[j], sel_destination[j])

directory_main = 'data/large/'
directories_destination = ['data/large/1/', 'data/large/2/', 'data/large/3/']

files = [f for f in listdir(directory_main) if isfile(join(directory_main, f))]
sel = [s for s in files if "tfrecords" in s]
random.shuffle(sel)
n_files = len(sel)
for i in range(3):
    sel_part = sel[np.int32(i*n_files/3):np.int32((i+1)*n_files/3)]
    sel_source = [directory_main + s for s in sel_part]
    sel_destination = [directories_destination[i] + s for s in sel_part]
    for j in range(len(sel_part)):
        shutil.move(sel_source[j], sel_destination[j])
