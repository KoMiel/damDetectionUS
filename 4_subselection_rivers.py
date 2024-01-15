
### this file ensures that there are roughly the same amount of negative and positive examples (images with dams and images without dams)



### import packages

import os
import shutil
from os.path import isfile, join
import random
from os import listdir



### import settings

with open('settings.json', 'r') as f:
  settings = json.load(f)

# primary parameters
directory = settings['directory']
storage_directory = settings['storage_directory'] # number of copies of negative locations (without a dam)



### function that calculates size of folder
def get_dir_size(path='.'):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total



### main code

random.seed(300) # random seed for reproducibility

# init
size_negatives = 1
size_positives = 0

# create output directories
try:
    os.mkdir(directory + storage_directory + '/rivers_reserve')
except:
    print('Directory already exists')

# directories
directory_rivers_reserve = 'data/rivers_reserve/'
directory_rivers = 'data/rivers/'

# get all river files
files = [f for f in listdir(directory_rivers) if isfile(join(directory_rivers, f))]
sel = [s for s in files if "tfrecords" in s]
# shuffle
random.shuffle(sel)

# counter
counter = 0

# loop that runs until size of files with dams exceeds size of files without dams
while size_negatives > size_positives:
    
    # get file
    sel_source = [directory_main + sel[counter]]
    # get destination
    sel_destination = [directories_destination[i] + sel[counter]]
    # move file
    shutil.move(sel_source, sel_destination)
    
    size_negatives = get_dir_size('data/rivers') + get_dir_size('data/lakes')
    size_positives = get_dir_size('data/very_small') + get_dir_size('data/small') + get_dir_size('data/medium') + get_dir_size('data/large')
