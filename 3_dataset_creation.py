
### in this script, earth engine data is combined with known locations of dams, rivers and lake pour points to create files that can be fed into CNNs in tensorflow/keras


### import packages

import random
import geopandas as gpd
import rasterio as rst
import pandas as pd
import numpy as np
import os
import json
import tensorflow



### import settings

with open('settings.json', 'r') as f:
  settings = json.load(f)

# primary parameters
directory = settings['directory']
layer_names = settings['layer_names']
resolution = settings['resolution'] # resolution for input
resolution_heatmap = settings['resolution_heatmap'] # resolution for intended output
median_dam_size = settings['median_dam_size'] # median dam size (calculated from data)
meter_feet_conversion_factor = settings['meter_feet_conversion_factor']
nCopies = settings['nCopies'] # number of copies for each dam (data augmentation)
pixels_image = settings['pixels_image'] # length of images (quadratic shape, pixels_image x pixels_image)
storage_directory = settings['storage_directory'] # number of copies of negative locations (without a dam)
n_images_shard = settings['n_images_shard'] # images per file (used for faster training)
num_negative_copies = settings['num_negative_copies'] # images per file (used for faster training)

# secondary parameters
n_layers = len(layer_names)
factor = resolution_heatmap/resolution # scaling between input and output
damSize = np.int32(np.ceil(median_dam_size/meter_feet_conversion_factor/resolution_heatmap/2)) # radius around dam locations (for heatmap approach with same sizes for all dams)
pixels_heatmap = int(pixels_image/factor)

random.seed(1) # random seed for reproducibility



### function used to serialize a single training example

def serialize_example(image,
                      heatmap_very_small,
                      heatmap_small,
                      heatmap_medium,
                      heatmap_large,
                      heatmap_all,
                      mercator_x,
                      mercator_y):
    # all variables apart from input images; one heatmap for each dam subset to be able to evaluate performance for each set
    feature_dict = {'heatmap/very_small': tensorflow.train.Feature(
                        bytes_list=tensorflow.train.BytesList(value=[heatmap_very_small.tobytes()])),
                    'heatmap/small': tensorflow.train.Feature(
                        bytes_list=tensorflow.train.BytesList(value=[heatmap_small.tobytes()])),
                    'heatmap/medium': tensorflow.train.Feature(
                        bytes_list=tensorflow.train.BytesList(value=[heatmap_medium.tobytes()])),
                    'heatmap/large': tensorflow.train.Feature(
                        bytes_list=tensorflow.train.BytesList(value=[heatmap_large.tobytes()])),
                    'heatmap/all': tensorflow.train.Feature(
                        bytes_list=tensorflow.train.BytesList(value=[heatmap_all.tobytes()])),
                    'image/mercator_x': tensorflow.train.Feature(float_list=tensorflow.train.FloatList(value=mercator_x)), # coordinates to be able to evaluate performance depending on geographical location
                    'image/mercator_y': tensorflow.train.Feature(float_list=tensorflow.train.FloatList(value=mercator_y))
                    }
    # normalization of images
    img_max = 10000
    img_min = 0
    image[0] = (image[0] - img_min)/(img_max - img_min)*2 - 1
    image[1] = (image[1] - img_min)/(img_max - img_min)*2 - 1
    image[2] = (image[2] - img_min)/(img_max - img_min)*2 - 1
    
    # normalization of elevation
    elev_max = 4421 # highest peak in US
    elev_min = 0
    image[3] = (image[3] - elev_min)/(elev_max - elev_min)*2 - 1
    
    # normalization of water occurrence
    water_max = 100
    water_min = 0
    image[4] = (image[4] - water_min)/(water_max - water_min)*2 - 1
    
    # add all layers
    for k in range(n_layers):
        layer = image[k]
        feature_dict['image/' + layer_names[k]] = tensorflow.train.Feature(
            bytes_list=tensorflow.train.BytesList(value=[layer.tobytes()]))
    example_proto = tensorflow.train.Example(features=tensorflow.train.Features(feature=feature_dict))
    return example_proto.SerializeToString() # return a single training example



### main code

# create output directories
try:
    os.mkdir(directory + storage_directory)
    os.mkdir(directory + storage_directory + '/dam_information')
    os.mkdir(directory + storage_directory + '/very_small/')
    os.mkdir(directory + storage_directory + '/very_small/1/')
    os.mkdir(directory + storage_directory + '/very_small/2/')
    os.mkdir(directory + storage_directory + '/very_small/3/')
    os.mkdir(directory + storage_directory + '/small/')
    os.mkdir(directory + storage_directory + '/small/1/')
    os.mkdir(directory + storage_directory + '/small/2/')
    os.mkdir(directory + storage_directory + '/small/3/')
    os.mkdir(directory + storage_directory + '/medium/')
    os.mkdir(directory + storage_directory + '/medium/1/')
    os.mkdir(directory + storage_directory + '/medium/2/')
    os.mkdir(directory + storage_directory + '/medium/3/')
    os.mkdir(directory + storage_directory + '/large/')
    os.mkdir(directory + storage_directory + '/large/1/')
    os.mkdir(directory + storage_directory + '/large/2/')
    os.mkdir(directory + storage_directory + '/large/3/')
    os.mkdir(directory + storage_directory + '/lakes/')
    os.mkdir(directory + storage_directory + '/lakes/1/')
    os.mkdir(directory + storage_directory + '/lakes/2/')
    os.mkdir(directory + storage_directory + '/lakes/3/')
    os.mkdir(directory + storage_directory + '/rivers/')
    os.mkdir(directory + storage_directory + '/rivers/1/')
    os.mkdir(directory + storage_directory + '/rivers/2/')
    os.mkdir(directory + storage_directory + '/rivers/3/')
except:
    print('Directory already exists')

# open US states geometries
states_shape = gpd.read_file('cb_2018_us_state_500k/')
states_shape = states_shape.to_crs('EPSG:3395')

# open us state information file
states_info = pd.read_csv('states.csv', delimiter='\t', dtype=str, header=0)

# read lakes file
lakes = pd.read_csv('lakes_latlong.csv', delimiter=',', dtype=float)
lakes = gpd.GeoDataFrame(lakes, geometry=gpd.points_from_xy(lakes.LATITUDE, lakes.LONGITUDE))
lakes = lakes.set_crs('EPSG:4326')
lakes = lakes.to_crs('EPSG:3395')

# read rivers file
rivers = pd.read_csv('rivers_latlong.csv', delimiter=',', dtype = float)
rivers = gpd.GeoDataFrame(rivers, geometry=gpd.points_from_xy(rivers.LATITUDE, rivers.LONGITUDE))
rivers = rivers.set_crs('EPSG:4326')
rivers = rivers.to_crs('EPSG:3395')

# get all filenames
files_in_dir = os.listdir('tif/')

# loop over all states
for i in range(50):

    # get information on current state
    state_id = states_info['Code '][i].strip()
    state_name = states_info['State '][i].strip()
    state_alpha = states_info['Alpha code'][i].strip()

    # print statement
    print(state_name)
    
    # no dams in Washington DC; skip
    if state_alpha == 'DC':
        continue
    
    # get shape of state
    state_shape = states_shape['geometry'][states_shape['STATEFP'] == state_id]

    # get all files belonging to current state
    state_files = [s for s in files_in_dir if state_name in s]

    # case distinction: one file or multiple files
    if len(state_files) > 1:
        
        # here, a single numpy array in which all image data from the state can fit in is created
        # get the last file to calculate the required size of numpy array
        last_file = directory + 'tif/' + sorted(state_files)[len(state_files) - 1]
        coord_range = last_file.split(sep='-')

        # get x and y coordinates of lower bottom corner
        x = int(coord_range[1])
        y = int(coord_range[2].strip('.tif'))
        
        # open file
        img = rst.open(last_file)
        dat = img.read()

        # get border coordinate
        y_min = img.bounds[1]

        # get shape of last file and add to previous sizes
        xAdd, yAdd = dat.shape[1:3]
        xComplete = x + xAdd
        yComplete = y + yAdd

        # generate empty array
        img_array = np.zeros(shape=(n_layers, xComplete, yComplete), dtype=np.float32)
        
        # loop over all files
        for count, file_name in enumerate(state_files):
            
            # open file
            filename = directory + 'tif/' + file_name
            img = rst.open(filename)
            dat = img.read()
            
            # get coordinates of lower bottom corner
            coords = file_name.split(sep='-')
            x = int(coords[1])
            y = int(coords[2].strip('.tif'))

            # get size of current file
            xAdd, yAdd = dat.shape[1:3]
            
            # insert data into image array
            if dat.shape[0] == 6:
                img_array[:, x:(x + xAdd), y:(y + yAdd)] = dat[[0, 1, 2, 3, 4]]
            else:
                img_array[:, x:(x + xAdd), y:(y + yAdd)] = dat[[0, 1, 2, 3, 4]]

        # get the first file to extract minimal x coordinates
        first_file = directory + 'tif/' + sorted(state_files)[0]

        # open file
        img_first = rst.open(first_file)

        # get coordinate
        x_min = img_first.bounds[0]

    elif len(state_files) == 1:

        # if only one file: just open it
        file = directory + 'tif/' + state_files[0]
        img = rst.open(file)
        img_array = img.read()
        
        # get coordinates
        x_min, y_min = img.bounds[0:2]
    
    else:
        continue
    
    # combine bounds
    bounds = np.array([x_min, y_min])

    # swap axes
    img_array = np.swapaxes(img_array, 1, 2)
    img_array = np.flip(img_array, 2)

    # replace NaNs in image files with 10000 (white for clouds)
    img_array[0:3][np.isnan(img_array)[0:3]] = 10000

    # replace NaNs with 0 (water/elevation)
    img_array[3:5][np.isnan(img_array)[3:5]] = 0

    # open dam file for current state
    dams_state = pd.read_excel(
        os.path.join('dams/' + state_alpha + '_U.xlsx'),
        engine='openpyxl'
    )
    
    # create geopandas dataframe
    dams_state = gpd.GeoDataFrame(
        dams_state, geometry=gpd.points_from_xy(dams_state.LONGITUDE, dams_state.LATITUDE))

    # convert to Mercator projection
    dams_state = dams_state.set_crs('EPSG:4326')
    dams_state = dams_state.to_crs('EPSG:3395')

    # create data frame for dam information in which to store supplementary information
    dam_information = pd.DataFrame(index=range(len(dams_state)), columns=['dam_size', 'dam_height', 'dam_storage', 'dam_lat', 'dam_long', 'dam_x', 'dam_y'])

    # create arrays for heatmaps
    dam_array_very_small = np.zeros(shape=[int(img_array.shape[1]/factor), int(img_array.shape[2]/factor)], dtype=np.float32)
    dam_array_small = np.zeros(shape=[int(img_array.shape[1]/factor), int(img_array.shape[2]/factor)], dtype=np.float32)
    dam_array_medium = np.zeros(shape=[int(img_array.shape[1]/factor), int(img_array.shape[2]/factor)], dtype=np.float32)
    dam_array_large = np.zeros(shape=[int(img_array.shape[1]/factor), int(img_array.shape[2]/factor)], dtype=np.float32)
    dam_array_all = np.zeros(shape=[int(img_array.shape[1]/factor), int(img_array.shape[2]/factor)],  dtype=np.float32)

    # loop over all dams
    for dam in range(len(dams_state)):
        
        # print statement
        print(dam)
        
        # check whether the dam truly belongs to current state
        inside = state_shape.contains(dams_state['geometry'].iloc[dam])
        if inside.any():
            
            # calculate x and y indices
            x_dam = round((dams_state['geometry'].iloc[dam].x - bounds[0])/resolution)
            y_dam = round((dams_state['geometry'].iloc[dam].y - bounds[1])/resolution)
            
            # calculate exact dam location
            x_ex = (dams_state['geometry'].iloc[dam].x - bounds[0])/resolution
            y_ex = (dams_state['geometry'].iloc[dam].y - bounds[1])/resolution
            
            # get supplementary information on dam
            dam_height = dams_state['NID_HEIGHT'].iloc[dam]/meter_feet_conversion_factor
            dam_size = dams_state['DAM_LENGTH'].iloc[dam]/meter_feet_conversion_factor
            dam_storage = dams_state['NID_STORAGE'].iloc[dam]
            
            # calculate the pixel radius of dam, if not available replace with median
            if np.isnan(dam_size):
                size = damSize
                size_ex = median_dam_size/meter_feet_conversion_factor/resolution_heatmap/2
            elif dam_size == 0:
                size = np.int32(1)
                size_ex = 1/meter_feet_conversion_factor/resolution_heatmap/2
            else:
                size = np.int32(np.ceil(dam_size/resolution_heatmap/2))
                size_ex = dam_size/resolution_heatmap/2
                
            # populate heatmaps, starting with heatmap for all dams
            for ind_x in range(-size, size + 1):
                for ind_y in range(-size, size + 1):
                    # calculate dam weight
                    dist = np.sqrt(((ind_x + (x_dam - x_ex)) * (ind_x + (x_dam - x_ex)) + (ind_y + (y_dam - y_ex)) * (ind_y + (y_dam - y_ex))))
                    if ind_x == 0 and ind_y == 0:
                        dist = 0
                    try:
                        # if value exceeds previous value of that location, take new value
                        dam_array_all[ind_x + round(x_dam/factor), ind_y + round(y_dam/factor)] = np.amax([np.exp(-(dist * dist) / (2 * size_ex / 3 * size_ex / 3)), dam_array_all[ind_x + round(x_dam/factor), ind_y + round(y_dam/factor)]])
                    except:
                        print('Outside of boundary')
            if np.isnan(dam_height):
                print('No size known')
            # repeat for subsets of dam heights
            elif dam_height < 5:
                for ind_x in range(-size, size + 1):
                    for ind_y in range(-size, size + 1):
                        dist = np.sqrt(((ind_x + (x_dam - x_ex)) * (ind_x + (x_dam - x_ex)) + (ind_y + (y_dam - y_ex)) * (ind_y + (y_dam - y_ex))))
                        if ind_x == 0 and ind_y == 0:
                            dist = 0
                        try:
                            dam_array_very_small[ind_x + round(x_dam/factor), ind_y + round(y_dam/factor)] = np.amax([np.exp(-(dist * dist) / (2 * size_ex / 3 * size_ex / 3)), dam_array_very_small[ind_x + round(x_dam/factor), ind_y + round(y_dam/factor)]])
                        except:
                            print('Outside of boundary')
            elif dam_height < 10:
                for ind_x in range(-size, size + 1):
                    for ind_y in range(-size, size + 1):
                        dist = np.sqrt(((ind_x + (x_dam - x_ex)) * (ind_x + (x_dam - x_ex)) + (ind_y + (y_dam - y_ex)) * (ind_y + (y_dam - y_ex))))
                        if ind_x == 0 and ind_y == 0:
                            dist = 0
                        try:
                            dam_array_small[ind_x + round(x_dam/factor), ind_y + round(y_dam/factor)] = np.amax([np.exp(-(dist * dist) / (2 * size_ex / 3 * size_ex / 3)), dam_array_small[ind_x + round(x_dam/factor), ind_y + round(y_dam/factor)]])
                        except:
                            print('Outside of boundary')
            elif dam_height < 15:
                for ind_x in range(-size, size + 1):
                    for ind_y in range(-size, size + 1):
                        dist = np.sqrt(((ind_x + (x_dam - x_ex)) * (ind_x + (x_dam - x_ex)) + (ind_y + (y_dam - y_ex)) * (ind_y + (y_dam - y_ex))))
                        if ind_x == 0 and ind_y == 0:
                            dist = 0
                        try:
                            dam_array_medium[ind_x + round(x_dam/factor), ind_y + round(y_dam/factor)] = np.amax([np.exp(-(dist * dist) / (2 * size_ex / 3 * size_ex / 3)), dam_array_medium[ind_x + round(x_dam/factor), ind_y + round(y_dam/factor)]])
                        except:
                            print('Outside of boundary')
            else:
                for ind_x in range(-size, size + 1):
                    for ind_y in range(-size, size + 1):
                        dist = np.sqrt(((ind_x + (x_dam - x_ex)) * (ind_x + (x_dam - x_ex)) + (ind_y + (y_dam - y_ex)) * (ind_y + (y_dam - y_ex))))
                        if ind_x == 0 and ind_y == 0:
                            dist = 0
                        try:
                            dam_array_large[ind_x + round(x_dam/factor), ind_y + round(y_dam/factor)] = np.amax([np.exp(-(dist * dist) / (2 * size_ex / 3 * size_ex / 3)), dam_array_large[ind_x + round(x_dam/factor), ind_y + round(y_dam/factor)]])
                        except:
                            print('Outside of boundary')
            # store information on dam
            dam_information.iloc[dam]['dam_size'] = dam_size
            dam_information.iloc[dam]['dam_height'] = dam_height
            dam_information.iloc[dam]['dam_storage'] = dam_storage
            dam_information.iloc[dam]['dam_lat'] = dams_state['LATITUDE'].iloc[dam]
            dam_information.iloc[dam]['dam_long'] = dams_state['LONGITUDE'].iloc[dam]
            dam_information.iloc[dam]['dam_x'] = dams_state['geometry'].x.iloc[dam]
            dam_information.iloc[dam]['dam_y'] = dams_state['geometry'].y.iloc[dam]

    # remove nans
    dam_information = dam_information[dam_information['dam_x'].notna()]
    dam_information = dam_information.sample(frac=1)
    dam_information.to_csv(directory + storage_directory + '/dam_information/' + state_name + '.csv')

    # create data frames for subsets
    very_small_dams = dam_information[dam_information['dam_height'] < 5]
    rest_dams = dam_information.drop(very_small_dams.index)
    small_dams = rest_dams[rest_dams['dam_height'] < 10]
    rest_dams = rest_dams.drop(small_dams.index)
    medium_dams = rest_dams[rest_dams['dam_height'] < 15]
    large_dams = rest_dams.drop(medium_dams.index)

    # create dataframe for lakes
    lake_information = pd.DataFrame(index=range(lakes.shape[0]), columns=['lake_x', 'lake_y'])
    
    # loop over lakes, if they are inside the current state add to dataframe
    for j in range(lakes.shape[0]):
        inside = state_shape.contains(lakes['geometry'].iloc[j])
        if inside.any():
            lake_information['lake_x'].iloc[j] = lakes['geometry'].iloc[j].x
            lake_information['lake_y'].iloc[j] = lakes['geometry'].iloc[j].y
            
    # remove nans
    lake_information = lake_information[lake_information['lake_x'].notna()]

    # repeat for rivers
    river_information = pd.DataFrame(index=range(rivers.shape[0]), columns=['river_x', 'river_y'])

    for j in range(rivers.shape[0]):
        inside = state_shape.contains(rivers['geometry'].iloc[j])
        if inside.any():
            river_information['river_x'].iloc[j] = rivers['geometry'].iloc[j].x
            river_information['river_y'].iloc[j] = rivers['geometry'].iloc[j].y
            
    river_information = river_information[river_information['river_x'].notna()]

    # calculate number of shards (files created for the CNN)
    n_shards = np.int32(np.ceil(very_small_dams.shape[0]/n_images_shard*nCopies))
    
    # number of dams
    n_dams = np.int32(very_small_dams.shape[0])
    
    # array for dam copy combinations
    dam_copy_id = np.repeat(range(n_dams), nCopies)

    # loop over shards (dams smaller 5m)
    for shard in range(n_shards):
        
        # index for first visited file
        index = shard * n_images_shard
        
        # print message
        print('shard number ', shard)
        
        # create filename
        shard_path = directory + storage_directory + '/very_small/' + state_alpha + '_' + '%.5d-of-%.5d' % (shard, n_shards - 1) + '.tfrecords'
        
        # find end file of shard (required because last shard will have less images)
        end = index + n_images_shard if dam_copy_id.shape[0] > (index + n_images_shard) else dam_copy_id.shape[0]
        
        # open tensorflow writer
        with tensorflow.io.TFRecordWriter(shard_path) as writer:
            
            # loop over images of shard
            for j in range(index, end):
                dam = dam_copy_id[j]
                x_dam = round((very_small_dams.iloc[dam]['dam_x'] - bounds[0])/resolution)
                y_dam = round((very_small_dams.iloc[dam]['dam_y'] - bounds[1])/resolution)
                
                # perform a random translation
                # it needs to be checked whether image has the right size after operations (reason for success check)
                success = False
                while success == False:
                    random_translation = np.random.randint(low=(-pixels_image/2/factor), high=(pixels_image/2/factor), size=2)
                    # crop image
                    image_cut_temp = img_array[:, np.int32(x_dam + random_translation[0]*factor - pixels_image/2):np.int32(x_dam + random_translation[0]*factor + pixels_image/2), np.int32(y_dam + random_translation[1]*factor - pixels_image/2):np.int32(y_dam + random_translation[1]*factor + pixels_image/2)]
                    if image_cut_temp.shape[1] == pixels_image and image_cut_temp.shape[2] == pixels_image:
                        success = True
                        
                
                image_cut = np.copy(image_cut_temp)
                
                # perform translation for all heatmaps as well
                dam_cut_very_small = np.copy(dam_array_very_small[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam + random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])
                dam_cut_small = np.copy(dam_array_small[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam + random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])
                dam_cut_medium = np.copy(dam_array_medium[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam + random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])
                dam_cut_large = np.copy(dam_array_large[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam +    random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])
                dam_cut_all = np.copy(dam_array_all[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam +    random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])

                # serialize example
                tf_example = serialize_example(image_cut, dam_cut_very_small, dam_cut_small, dam_cut_medium, dam_cut_large, dam_cut_all, [very_small_dams.iloc[dam]['dam_x'] + random_translation[0]*factor*resolution - pixels_image/2*resolution], [very_small_dams.iloc[dam]['dam_y'] + random_translation[1]*factor*resolution - pixels_image/2*resolution])
                writer.write(tf_example)



    # repeat for dams 5 to smaller than 10
    n_shards = np.int32(np.ceil(small_dams.shape[0]/n_images_shard*nCopies))
    n_dams = np.int32(small_dams.shape[0])
    dam_copy_id = np.repeat(range(n_dams), nCopies)

    for shard in range(n_shards):
        index = shard * n_images_shard
        print('shard number ', shard)
        shard_path = directory + storage_directory + '/small/' + state_alpha + '_' + '%.5d-of-%.5d' % (shard, n_shards - 1) + '.tfrecords'
        end = index + n_images_shard if dam_copy_id.shape[0] > (index + n_images_shard) else dam_copy_id.shape[0]

        with tensorflow.io.TFRecordWriter(shard_path) as writer:
            for j in range(index, end):
                dam = dam_copy_id[j]
                x_dam = round((small_dams.iloc[dam]['dam_x'] - bounds[0])/resolution)
                y_dam = round((small_dams.iloc[dam]['dam_y'] - bounds[1])/resolution)
                success = False
                while success == False:
                    random_translation = np.random.randint(low=(-pixels_image/2/factor), high=(pixels_image/2/factor), size=2)
                    image_cut_temp = img_array[:, np.int32(x_dam + random_translation[0]*factor - pixels_image/2):np.int32(x_dam + random_translation[0]*factor + pixels_image/2), np.int32(y_dam + random_translation[1]*factor - pixels_image/2):np.int32(y_dam + random_translation[1]*factor + pixels_image/2)]
                    if image_cut_temp.shape[1] == pixels_image and image_cut_temp.shape[2] == pixels_image:
                        success = True
                        
                image_cut = np.copy(image_cut_temp)
                dam_cut_very_small = np.copy(dam_array_very_small[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam + random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])
                dam_cut_small = np.copy(dam_array_small[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam + random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])
                dam_cut_medium = np.copy(dam_array_medium[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam + random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])
                dam_cut_large = np.copy(dam_array_large[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam +    random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])
                dam_cut_all = np.copy(dam_array_all[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam +    random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])

                tf_example = serialize_example(image_cut, dam_cut_very_small, dam_cut_small, dam_cut_medium, dam_cut_large, dam_cut_all, [small_dams.iloc[dam]['dam_x'] + random_translation[0]*factor*resolution - pixels_image/2*resolution], [small_dams.iloc[dam]['dam_y'] + random_translation[1]*factor*resolution - pixels_image/2*resolution])
                writer.write(tf_example)
    


    # repeat for dams 10 to smaller than 15
    n_shards = np.int32(np.ceil(medium_dams.shape[0]/n_images_shard*nCopies))
    n_dams = np.int32(medium_dams.shape[0])
    dam_copy_id = np.repeat(range(n_dams), nCopies)

    for shard in range(n_shards):
        index = shard * n_images_shard
        print('shard number ', shard)
        shard_path = directory + storage_directory + '/medium/' + state_alpha + '_' + '%.5d-of-%.5d' % (shard, n_shards - 1) + '.tfrecords'
        end = index + n_images_shard if dam_copy_id.shape[0] > (index + n_images_shard) else dam_copy_id.shape[0]

        with tensorflow.io.TFRecordWriter(shard_path) as writer:
            for j in range(index, end):
                dam = dam_copy_id[j]
                x_dam = round((medium_dams.iloc[dam]['dam_x'] - bounds[0])/resolution)
                y_dam = round((medium_dams.iloc[dam]['dam_y'] - bounds[1])/resolution)
                success = False
                while success == False:
                    random_translation = np.random.randint(low=(-pixels_image/2/factor), high=(pixels_image/2/factor), size=2)
                    image_cut_temp = img_array[:, np.int32(x_dam + random_translation[0]*factor - pixels_image/2):np.int32(x_dam + random_translation[0]*factor + pixels_image/2), np.int32(y_dam + random_translation[1]*factor - pixels_image/2):np.int32(y_dam + random_translation[1]*factor + pixels_image/2)]
                    if image_cut_temp.shape[1] == pixels_image and image_cut_temp.shape[2] == pixels_image:
                        success = True
                        
                image_cut = np.copy(image_cut_temp)
                dam_cut_very_small = np.copy(dam_array_very_small[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam + random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])
                dam_cut_small = np.copy(dam_array_small[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam + random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])
                dam_cut_medium = np.copy(dam_array_medium[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam + random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])
                dam_cut_large = np.copy(dam_array_large[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam +    random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])
                dam_cut_all = np.copy(dam_array_all[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam +    random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])

                tf_example = serialize_example(image_cut, dam_cut_very_small, dam_cut_small, dam_cut_medium, dam_cut_large, dam_cut_all, [medium_dams.iloc[dam]['dam_x'] + random_translation[0]*factor*resolution - pixels_image/2*resolution], [medium_dams.iloc[dam]['dam_y'] + random_translation[1]*factor*resolution - pixels_image/2*resolution])
                writer.write(tf_example)



    # repeat for dams 15 and larger
    n_shards = np.int32(np.ceil(large_dams.shape[0]/n_images_shard*nCopies))
    n_dams = np.int32(large_dams.shape[0])
    dam_copy_id = np.repeat(range(n_dams), nCopies)

    for shard in range(n_shards):
        index = shard * n_images_shard
        print('shard number ', shard)
        shard_path = directory + storage_directory + '/large/' + state_alpha + '_' + '%.5d-of-%.5d' % (shard, n_shards - 1) + '.tfrecords'
        end = index + n_images_shard if dam_copy_id.shape[0] > (index + n_images_shard) else dam_copy_id.shape[0]
        
        with tensorflow.io.TFRecordWriter(shard_path) as writer:
            for j in range(index, end):
                dam = dam_copy_id[j]
                x_dam = round((large_dams.iloc[dam]['dam_x'] - bounds[0])/resolution)
                y_dam = round((large_dams.iloc[dam]['dam_y'] - bounds[1])/resolution)
                success = False
                while success == False:
                    random_translation = np.random.randint(low=(-pixels_image/2/factor), high=(pixels_image/2/factor), size=2)
                    image_cut_temp = img_array[:, np.int32(x_dam + random_translation[0]*factor - pixels_image/2):np.int32(x_dam + random_translation[0]*factor + pixels_image/2), np.int32(y_dam + random_translation[1]*factor - pixels_image/2):np.int32(y_dam + random_translation[1]*factor + pixels_image/2)]
                    if image_cut_temp.shape[1] == pixels_image and image_cut_temp.shape[2] == pixels_image:
                        success = True
                        
                image_cut = np.copy(image_cut_temp)
                dam_cut_very_small = np.copy(dam_array_very_small[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam + random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])
                dam_cut_small = np.copy(dam_array_small[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam + random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])
                dam_cut_medium = np.copy(dam_array_medium[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam + random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])
                dam_cut_large = np.copy(dam_array_large[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam +    random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])
                dam_cut_all = np.copy(dam_array_all[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam +    random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])

                tf_example = serialize_example(image_cut, dam_cut_very_small, dam_cut_small, dam_cut_medium, dam_cut_large, dam_cut_all, [large_dams.iloc[dam]['dam_x'] + random_translation[0]*factor*resolution - pixels_image/2*resolution], [large_dams.iloc[dam]['dam_y'] + random_translation[1]*factor*resolution - pixels_image/2*resolution])
                writer.write(tf_example)

                
              
    
            
    # repeat for lakes
    n_shards = np.int32(np.ceil(lake_information.shape[0]/n_images_shard*nCopies))
    n_dams = np.int32(lake_information.shape[0])
    dam_copy_id = np.repeat(range(n_dams), nCopies)

    for shard in range(n_shards):
        index = shard * n_images_shard
        print('shard number ', shard)
        shard_path = directory + storage_directory + '/lakes/' + state_alpha + '_' + '%.5d-of-%.5d' % (shard, n_shards - 1) + '.tfrecords'
        end = index + n_images_shard if dam_copy_id.shape[0] > (index + n_images_shard) else dam_copy_id.shape[0]

        with tensorflow.io.TFRecordWriter(shard_path) as writer:
            for j in range(index, end):
                # try ten times; if is hasn't found a fitting option then, go to next
                counter = 0
                dam = dam_copy_id[j]
                x_dam = round((lake_information.iloc[dam]['lake_x'] - bounds[0])/resolution)
                y_dam = round((lake_information.iloc[dam]['lake_y'] - bounds[1])/resolution)
                success = False
                while success == False:
                    random_translation = np.random.randint(low=(-pixels_image/2/factor), high=(pixels_image/2/factor), size=2)
                    image_cut_temp = img_array[:, np.int32(x_dam + random_translation[0]*factor - pixels_image/2):np.int32(x_dam + random_translation[0]*factor + pixels_image/2), np.int32(y_dam + random_translation[1]*factor - pixels_image/2):np.int32(y_dam + random_translation[1]*factor + pixels_image/2)]
                    dam_cut_all_temp = np.copy(dam_array_all[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam +    random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])
                    counter += 1
                    if image_cut_temp.shape[1] == pixels_image and image_cut_temp.shape[2] == pixels_image and np.amax(dam_cut_all_temp) < 1:
                        success = True
                    elif counter == num_negative_copies:
                        break
                if counter == num_negative_copies and success == False:
                    continue
                
                image_cut = np.copy(image_cut_temp)
                
                dam_cut_very_small = np.copy(dam_array_very_small[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam + random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])
                dam_cut_small = np.copy(dam_array_small[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam + random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])
                dam_cut_medium = np.copy(dam_array_medium[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam + random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])
                dam_cut_large = np.copy(dam_array_large[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam +    random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])
                dam_cut_all = np.copy(dam_array_all[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam +    random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])

                tf_example = serialize_example(image_cut, dam_cut_very_small, dam_cut_small, dam_cut_medium, dam_cut_large, dam_cut_all, [lake_information.iloc[dam]['lake_x'] + random_translation[0]*factor*resolution - pixels_image/2*resolution], [lake_information.iloc[dam]['lake_y'] + random_translation[1]*factor*resolution - pixels_image/2*resolution])
                writer.write(tf_example)


        
    # repeat for rivers
    n_shards = np.int32(np.ceil(river_information.shape[0]/n_images_shard*nCopies))
    n_dams = np.int32(river_information.shape[0])
    dam_copy_id = np.repeat(range(n_dams), nCopies)

    for shard in range(n_shards):
        index = shard * n_images_shard
        print('shard number ', shard)
        shard_path = directory + storage_directory + '/rivers/' + state_alpha + '_' + '%.5d-of-%.5d' % (shard, n_shards - 1) + '.tfrecords'
        end = index + n_images_shard if dam_copy_id.shape[0] > (index + n_images_shard) else dam_copy_id.shape[0]
        
        with tensorflow.io.TFRecordWriter(shard_path) as writer:
            for j in range(index, end):
                counter = 0
                dam = dam_copy_id[j]
                x_dam = round((river_information.iloc[dam]['river_x'] - bounds[0])/resolution)
                y_dam = round((river_information.iloc[dam]['river_y'] - bounds[1])/resolution)
                success = False
                while success == False:
                    random_translation = np.random.randint(low=(-pixels_image/2/factor), high=(pixels_image/2/factor), size=2)
                    image_cut_temp = img_array[:, np.int32(x_dam + random_translation[0]*factor - pixels_image/2):np.int32(x_dam + random_translation[0]*factor + pixels_image/2), np.int32(y_dam + random_translation[1]*factor - pixels_image/2):np.int32(y_dam + random_translation[1]*factor + pixels_image/2)]
                    dam_cut_all_temp = np.copy(dam_array_all[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam +    random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])
                    counter += 1
                    if image_cut_temp.shape[1] == pixels_image and image_cut_temp.shape[2] == pixels_image and np.amax(dam_cut_all_temp) < 1:
                        success = True
                    elif counter == num_negative_copies:
                        break
                if counter == num_negative_copies and success == False:
                    continue
                
                image_cut = np.copy(image_cut_temp)
                
                dam_cut_very_small = np.copy(dam_array_very_small[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam + random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])
                dam_cut_small = np.copy(dam_array_small[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam + random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])
                dam_cut_medium = np.copy(dam_array_medium[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam + random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])
                dam_cut_large = np.copy(dam_array_large[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam +    random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])
                dam_cut_all = np.copy(dam_array_all[np.int32((x_dam + random_translation[0]*factor - pixels_image/2)/factor):np.int32((x_dam + random_translation[0]*factor + pixels_image/2)/factor), np.int32((y_dam +    random_translation[1]*factor - pixels_image/2)/factor):np.int32((y_dam + random_translation[1]*factor + pixels_image/2)/factor)])

                tf_example = serialize_example(image_cut, dam_cut_very_small, dam_cut_small, dam_cut_medium, dam_cut_large, dam_cut_all, [river_information.iloc[dam]['river_x'] + random_translation[0]*factor*resolution - pixels_image/2*resolution], [river_information.iloc[dam]['river_y'] + random_translation[1]*factor*resolution - pixels_image/2*resolution])
                writer.write(tf_example)

    # delete arrays before continuing with next state
    del img_array
    del dam_array_small
    del dam_array_very_small
    del dam_array_medium
    del dam_array_large
    del dam_array_all
