
### this script trains a CNN and applies it to unseen data
### example file for one cross validation fold and all dam data



### import packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, UpSampling2D, Add, Dropout
from tensorflow.keras.models import Model
import keras.backend as K

import numpy as np
import random

from os import listdir
from os.path import isfile, join


random.seed(1000) # random seed for reproducibility



### set parameters

with open('settings.json', 'r') as f:
  settings = json.load(f)

model_name = 'a_1' # for all dams (for other models: a_2, a_3, vs_1, vs_2, vs_3, s_1, s_2, s_3, m_1, m_2, m_3, l_1, l_2, l_3)
g_truth = 'all' # take heatmap with all dams (for other models: all, all, very_small, very_small, very_small, small, small, small, medium, medium, medium, large, large, large)

# directories with data used for training/model selection (for other cross validation folds, shuffle around)
data_directory_training = ['data2/very_small/1/', 'data2/very_small/2/',
                           'data2/small/1/', 'data2/small/2/',
                           'data2/medium/1/', 'data2/medium/2/',
                           'data2/large/1/', 'data2/large/2/']
negative_directory_training = ['data2/lakes/1/', 'data2/lakes/2/',
                            'data2/rivers/1/', 'data2/rivers/2/']

# directories with data used for testing (for other cross validation folds, shuffle around)
data_directory_testing = ['data2/lakes/3/',
                           'data2/rivers/3/',
                           'data2/very_small/3/',
                           'data2/small/3/',
                           'data2/medium/3/',
                           'data2/large/3/']



# for evaluation: map the datasets to the types of dams associated with them (lakes, rivers = no dams)
g_truth_application = ['none',
                       'none',
                       'verysmall',
                       'small',
                       'medium',
                       'large']

# for evaluation, short name on where to store files
output_short = ['lakes',
                'rivers',
                'very_small',
                'small',
                'medium',
                'large']

# get layer names
layer_names = settings['layer_names']
resolution = settings['resolution'] # resolution for input
resolution_heatmap = settings['resolution_heatmap'] # resolution for intended output

factor = resolution_heatmap/resolution # scaling between input and output

size_input = settings['pixels_image'] # size of input
size_label = int(size_input/factor) # size of output

size_batch = settings['size_batch'] # batch size
learning_rate = settings['learning_rate'] # learning rate
num_epochs = settings['num_epochs'] # number of epochs

kernel_size = settings['kernel_size'] # size of kernel in middle part of the network (main part)
kernel_size_first = settings['kernel_size_first'] # size of kernel in first layer of network
kernel_size_last = settings['kernel_size_last'] # size of kernel in later stages of network
dropout_rate = settings['dropout_rate'] # rate for dropout of neurons
filter_multiplication_factor = settings['filter_multiplication_factor'] # a factor used to reduce number of filters in the definition of the network (this factor is required a high number of times)
split_train = settings['split_train'] # fraction of data used for training
split_val = settings['split_val'] # fraction of data used for model validation

output_directory = settings['output_directory']
model_directory = settings['model_directory']

num_layers = len(layer_names)
shape_input = (size_input, size_input, num_layers) # input shape used for setting up the model
size_shuffle_buffer = 25*size_batch # size of the shuffle buffer (the higher, the better the shuffling before training, but increases training time needed)



### function to parse single images for all dams data (one function per data type)

def _parse_image_function_all(example):

    # create dict for features, starting with heatmap
    feature_dict = {'heatmap/all': tf.io.FixedLenFeature(shape=[], dtype=tf.string)}

    # add input layers
    for layer in layer_names:
        feature_dict['image/' + layer] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)

    # parse single example
    features = tf.io.parse_single_example(example, feature_dict)

    # vector for layers
    imgs = []

    # combine input layers
    for layer in layer_names:
        image = tf.io.decode_raw(features['image/' + layer], out_type=tf.float32)
        image = tf.reshape(image, [size_input, size_input])
        imgs.append(image)

    # stack together
    imgs = tf.stack(imgs, 2)

    # decode label
    label = tf.io.decode_raw(features['heatmap/all'], out_type=tf.float32)

    # get label 'image'
    label = tf.reshape(label, [size_label, size_label])

    # return
    return imgs, label



### same function for all other heatmaps

def _parse_image_function_verysmall(example):

    feature_dict = {'heatmap/very_small': tf.io.FixedLenFeature(shape=[], dtype=tf.string)}

    for layer in layer_names:
        feature_dict['image/' + layer] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)

    features = tf.io.parse_single_example(example, feature_dict)

    imgs = []

    for layer in layer_names:
        image = tf.io.decode_raw(features['image/' + layer], out_type=tf.float32)
        image = tf.reshape(image, [size_input, size_input])
        imgs.append(image)

    imgs = tf.stack(imgs, 2)

    label = tf.io.decode_raw(features['heatmap/very_small'], out_type=tf.float32)

    label = tf.reshape(label, [size_label, size_label])

    return imgs, label



def _parse_image_function_small(example):

    feature_dict = {'heatmap/small': tf.io.FixedLenFeature(shape=[], dtype=tf.string)}

    for layer in layer_names:
        feature_dict['image/' + layer] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)

    features = tf.io.parse_single_example(example, feature_dict)

    imgs = []

    for layer in layer_names:
        image = tf.io.decode_raw(features['image/' + layer], out_type=tf.float32)
        image = tf.reshape(image, [size_input, size_input])
        imgs.append(image)

    imgs = tf.stack(imgs, 2)

    label = tf.io.decode_raw(features['heatmap/small'], out_type=tf.float32)

    label = tf.reshape(label, [size_label, size_label])

    return imgs, label



def _parse_image_function_medium(example):

    feature_dict = {'heatmap/medium': tf.io.FixedLenFeature(shape=[], dtype=tf.string)}

    for layer in layer_names:
        feature_dict['image/' + layer] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)

    features = tf.io.parse_single_example(example, feature_dict)

    imgs = []

    for layer in layer_names:
        image = tf.io.decode_raw(features['image/' + layer], out_type=tf.float32)
        image = tf.reshape(image, [size_input, size_input])
        imgs.append(image)

    imgs = tf.stack(imgs, 2)

    label = tf.io.decode_raw(features['heatmap/medium'], out_type=tf.float32)

    label = tf.reshape(label, [size_label, size_label])

    return imgs, label



def _parse_image_function_large(example):

    feature_dict = {'heatmap/large': tf.io.FixedLenFeature(shape=[], dtype=tf.string)}

    for layer in layer_names:
        feature_dict['image/' + layer] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)

    features = tf.io.parse_single_example(example, feature_dict)

    imgs = []

    for layer in layer_names:
        image = tf.io.decode_raw(features['image/' + layer], out_type=tf.float32)
        image = tf.reshape(image, [size_input, size_input])
        imgs.append(image)

    imgs = tf.stack(imgs, 2)

    label = tf.io.decode_raw(features['heatmap/large'], out_type=tf.float32)

    label = tf.reshape(label, [size_label, size_label])

    return imgs, label


# function to parse single images and get their mercator coordinates

def _parse_image_function_mercator(example):

    # start feature dict with mercator coordinates
    feature_dict = {'image/mercator_x': tf.io.FixedLenFeature(shape=[1], dtype = tf.float32),
                    'image/mercator_y': tf.io.FixedLenFeature(shape=[1], dtype = tf.float32)
    }

    # parse single example
    features = tf.io.parse_single_example(example, feature_dict)

    # get variables
    image_mercator_x = features['image/mercator_x']
    image_mercator_y = features['image/mercator_y']

    # return
    return image_mercator_x, image_mercator_y



# function to read datasets

def read_dataset(filenames, ground_truth, shuffle, epochs):

    # shuffle filenames
    if shuffle:
        random.shuffle(filenames)
    
    # open files
    shards = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = shards.interleave(tf.data.TFRecordDataset, cycle_length = 4)

    # if training, repeat and shuffle
    if shuffle:
        dataset = dataset.repeat(epochs).shuffle(buffer_size=size_shuffle_buffer)

    # get requested ground truth, according to parsing function 
    if ground_truth == 'all':
        dataset = dataset.map(_parse_image_function_all, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif ground_truth == 'verysmall':
        dataset = dataset.map(_parse_image_function_verysmall, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif ground_truth == 'small':
        dataset = dataset.map(_parse_image_function_small, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif ground_truth == 'medium':
        dataset = dataset.map(_parse_image_function_medium, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif ground_truth == 'large':
        dataset = dataset.map(_parse_image_function_large, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif ground_truth == 'mercator':
        dataset = dataset.map(_parse_image_function_mercator, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # create batch
    dataset = dataset.batch(size_batch)

    # prefetch data
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    # return dataset
    return dataset



### function that defines binary focal loss

def binary_focal_loss(gamma=2., alpha=.25, delta = 4.):
    def binary_focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # Clip the prediction value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # calculate weighting factor
        weight2 = tf.where(K.equal(y_true, 1), 1.0, K.pow((1 - y_true), delta))
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * weight2 * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss
    return binary_focal_loss_fixed



# function to combine relu activation and batch normalization

def bn_relu(inputs):

    # batch normalization
    bn = BatchNormalization()(inputs)

    # relu activation
    relu = ReLU()(bn)

    # return
    return relu



# a function that creates a residual layer for resnet approach

def resBlock(layer_in, filters, stride, flag):

    # first layer
    a = Conv2D(kernel_size=kernel_size,
               strides=stride,
               filters=filters,
               padding="same")(layer_in)
    a = bn_relu(a)
    
    # second layer
    a = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(a)
    a = bn_relu(a)

    # case distinction, based on where the layer is placed in the network
    if stride > 1 or flag == True:
        b = Conv2D(kernel_size=kernel_size,
                strides=stride,
                filters=filters,
                padding="same")(layer_in)
        b = bn_relu(b)
    else:
        b = layer_in

    # sum output of layers
    c = Add()([a, b])

    # return
    return c



### downsampling layer for upper part of network

def Downsample(layer_in, filters, flag = False):

    # resblock
    a = resBlock(layer_in = layer_in, filters = filters, stride = 2, flag = flag)
    
    # apply dropout
    a = Dropout(dropout_rate)(a)
    
    # second resblock
    a = resBlock(layer_in = a, filters = filters, stride = 1, flag = flag)
    
    # return
    return a



### upsampling layer for lower part of network

def Upsample(layer_in, filters, flag = False):

    # first resblock
    a = resBlock(layer_in = layer_in, filters = filters, stride = 1, flag = flag)
    
    # apply dropout
    a = Dropout(dropout_rate)(a)
    
    # second resblock
    a = resBlock(layer_in = a, filters = filters, stride = 1, flag = flag)
    
    # apply dropout
    a = Dropout(dropout_rate)(a)
    
    # upample to increase window size
    a = UpSampling2D()(a)

    # return
    return a



### connection layer to connect upsampling and downsampling parts of network

def Connect(layer_in, filters, flag = False):

    # resblock
    a = resBlock(layer_in = layer_in, filters = filters, stride = 1, flag = flag)
    
    # apply dropout
    a = Dropout(dropout_rate)(a)

    # return
    return a



### function that defines the hourglass network

def hourglass():

    # define input
    inputs = Input(shape=shape_input)

    # first layers
    start1 = Conv2D(kernel_size=kernel_size_first,
                strides=4,
                filters=16*filter_multiplication_factor,
                padding="same")(inputs)
    start2 = bn_relu(start1)
    start3 = resBlock(layer_in = start2, filters = 12*filter_multiplication_factor, stride = 1, flag = True)

    # downsampling layers
    feature1 = Downsample(layer_in = start3, filters = 16*filter_multiplication_factor) # 56
    feature2 = Downsample(layer_in = feature1, filters = 16*filter_multiplication_factor) # 28
    feature3 = Downsample(layer_in = feature2, filters = 24*filter_multiplication_factor) # 14
    feature4 = Downsample(layer_in = feature3, filters = 32*filter_multiplication_factor) # 7

    # middle of the network
    feature5 = Connect(layer_in = feature4, filters = 32*filter_multiplication_factor) # 7
    feature5 = Connect(layer_in = feature5, filters = 32*filter_multiplication_factor) # 7
    feature5 = Connect(layer_in = feature5, filters = 32*filter_multiplication_factor) # 7
    feature5 = Connect(layer_in = feature5, filters = 32*filter_multiplication_factor) # 7
    feature5 = Connect(layer_in = feature5, filters = 32*filter_multiplication_factor) # 7
    
    # connecting layers
    skip0 = Conv2D(kernel_size=kernel_size,
                   strides=1,
                   filters=12*filter_multiplication_factor,
                   padding="same")(start3)
    skip0 = bn_relu(skip0)
    skip1 = Connect(layer_in = start3, filters = 16*filter_multiplication_factor, flag = True)
    skip1 = Connect(layer_in = skip1, filters = 16*filter_multiplication_factor)
    skip2 = Connect(layer_in = feature1, filters = 16*filter_multiplication_factor)
    skip2 = Connect(layer_in = skip2, filters = 16*filter_multiplication_factor)
    skip3 = Connect(layer_in = feature2, filters = 16*filter_multiplication_factor)
    skip3 = Connect(layer_in = skip3, filters = 16*filter_multiplication_factor)
    skip4 = Connect(layer_in = feature3, filters = 24*filter_multiplication_factor)
    skip4 = Connect(layer_in = skip4, filters = 24*filter_multiplication_factor)

    # upsampling layers
    feature6 = Upsample(layer_in = feature5, filters = 24*filter_multiplication_factor, flag = True)
    feature7 = Add()([feature6, skip4])
    feature7 = Upsample(layer_in = feature6, filters = 16*filter_multiplication_factor, flag = True)
    feature8 = Add()([feature7, skip3])
    feature8 = Upsample(layer_in = feature7, filters = 16*filter_multiplication_factor)
    feature9 = Add()([feature8, skip2])
    feature9 = Upsample(layer_in = feature8, filters = 16*filter_multiplication_factor)
    feature10 = Add()([feature9, skip1])

    # final layers of first hourglass
    end1 = Conv2D(kernel_size=kernel_size,
                  strides=1,
                  filters=12*filter_multiplication_factor,
                  padding="same")(feature10)
    end1 = bn_relu(end1)
    end2 = Conv2D(kernel_size=kernel_size_last,
                  strides=1,
                  filters=12*filter_multiplication_factor,
                  padding="same")(end1)
    end2 = bn_relu(end2)
    end3 = Add()([end2, skip0])
    end4 = resBlock(layer_in = end3, filters = 12*filter_multiplication_factor, stride = 1, flag = False)
    end4 = resBlock(layer_in = end4, filters = 12*filter_multiplication_factor, stride = 1, flag = False)
    
    # downsampling layers
    feature11 = Downsample(layer_in = end4, filters = 16*filter_multiplication_factor) # 56
    feature12 = Downsample(layer_in = feature11, filters = 16*filter_multiplication_factor) # 28
    feature13 = Downsample(layer_in = feature12, filters = 24*filter_multiplication_factor) # 14
    feature14 = Downsample(layer_in = feature13, filters = 32*filter_multiplication_factor) # 7

    # middle of the network layers
    feature15 = Connect(layer_in = feature14, filters = 32*filter_multiplication_factor) # 7
    feature15 = Connect(layer_in = feature15, filters = 32*filter_multiplication_factor) # 7
    feature15 = Connect(layer_in = feature15, filters = 32*filter_multiplication_factor) # 7
    feature15 = Connect(layer_in = feature15, filters = 32*filter_multiplication_factor) # 7
    feature15 = Connect(layer_in = feature15, filters = 32*filter_multiplication_factor) # 7
    
    # connecting layers
    skip10 = Conv2D(kernel_size=kernel_size,
                   strides=1,
                   filters=12*filter_multiplication_factor,
                   padding="same")(end4)
    skip10 = bn_relu(skip0)
    skip11 = Connect(layer_in = end4, filters = 16*filter_multiplication_factor, flag = True)
    skip11 = Connect(layer_in = skip11, filters = 16*filter_multiplication_factor)
    skip12 = Connect(layer_in = feature11, filters = 16*filter_multiplication_factor)
    skip12 = Connect(layer_in = skip12, filters = 16*filter_multiplication_factor)
    skip13 = Connect(layer_in = feature12, filters = 16*filter_multiplication_factor)
    skip13 = Connect(layer_in = skip13, filters = 16*filter_multiplication_factor)
    skip14 = Connect(layer_in = feature13, filters = 24*filter_multiplication_factor)
    skip14 = Connect(layer_in = skip14, filters = 24*filter_multiplication_factor)

    # upsampling layers
    feature16 = Upsample(layer_in = feature15, filters = 24*filter_multiplication_factor, flag = True)
    feature17 = Add()([feature16, skip14])
    feature17 = Upsample(layer_in = feature16, filters = 16*filter_multiplication_factor, flag = True)
    feature18 = Add()([feature17, skip13])
    feature18 = Upsample(layer_in = feature17, filters = 16*filter_multiplication_factor)
    feature19 = Add()([feature18, skip12])
    feature19 = Upsample(layer_in = feature18, filters = 16*filter_multiplication_factor)
    feature20 = Add()([feature19, skip11])

    # final layers of the second hourglass
    end11 = Conv2D(kernel_size=kernel_size,
                  strides=1,
                  filters=12*filter_multiplication_factor,
                  padding="same")(feature20)
    end11 = bn_relu(end11)
    end12 = Conv2D(kernel_size=kernel_last,
                  strides=1,
                  filters=12*filter_multiplication_factor,
                  padding="same")(end11)
    end12 = bn_relu(end12)
    end13 = Add()([end12, skip10])
    end14 = resBlock(layer_in = end13, filters = 12*filter_multiplication_factor, stride = 1, flag = False)
    end14 = resBlock(layer_in = end14, filters = 12*filter_multiplication_factor, stride = 1, flag = False)
    end15 = Conv2D(kernel_size=kernel_last,
                  strides=1,
                  filters=1,
                  padding="same")(end14)

    # outputs
    outputs = keras.layers.Activation('sigmoid')(end15)

    # create model
    mod = Model(inputs, outputs)

    # compile model
    mod.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=binary_focal_loss())

    # return
    return mod



### function to train a model

def train_model(training_directories, negative_training_directories, ground_truth, epochs, new_model = True):

    # empty list for filenames with dams
    filenames = []
    
    # get all filenames and put into list
    for direct in training_directories:
        files = [f for f in listdir(direct) if isfile(join(direct, f))]
        sel = [s for s in files if "tfrecords" in s]
        sel = [direct + s for s in sel]
        selection_array = np.array(sel)
        filenames = np.append(filenames, selection_array)

    # empty list for filenames without dams
    filenames_negative = []

    # get all filenames and put into list
    for direct in negative_training_directories:
        files = [f for f in listdir(direct) if isfile(join(direct, f))]
        sel = [s for s in files if "tfrecords" in s]
        sel = [direct + s for s in sel]
        selection_array = np.array(sel)
        filenames_negative = np.append(filenames_negative, selection_array)
        
    # shuffle filenames
    random.shuffle(filenames)
    random.shuffle(filenames_negative)

    # split data into training and validation
    nTrain = int(round(split_train * filenames.shape[0]))
    nTrainNeg = int(round(split_train * filenames_negative.shape[0]))
    files_train = np.concatenate([filenames[0:nTrain], filenames_negative[0:nTrainNeg]])
    files_val = np.concatenate([filenames[nTrain:], filenames_negative[nTrainNeg:]])
    
    # calculate required number of training steps per epoch
    train_steps = int(round(len(files_train)*per_file/size_batch))
    
    # initiate model
    model = hourglass()
    
    # print statement
    print(model.summary())

    # read datasets
    dataset_train = read_dataset(files_train, ground_truth = ground_truth, shuffle = True, epochs = epochs)
    dataset_val = read_dataset(files_val, ground_truth = ground_truth, shuffle = False, epochs = 1)

    # read weights (if the model is not trained from scratch)
    if not new_model:
        model.load_weights('models/' + model_name + '.keras')

    # define callbacks for performance monitoring
    filepath_checkpoint = 'models/' + model_name + '.keras'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath_checkpoint,
        save_weights_only=True,
        monitor='val_loss',
        save_best_only=True)

    # fit the model
    model.fit(dataset_train,
                  validation_data=dataset_val,
                  epochs=epochs,
                  steps_per_epoch=train_steps,
                  batch_size=size_batch,
                  callbacks=[model_checkpoint_callback]
                  )



### apply the model and save the results

def apply_model(testing_directories, output_path, ground_truth, files_at_once = 10):

    # empty list for filenames
    filenames = []

    # get all filenames that are requested
    for direct in testing_directories:
        files = [f for f in listdir(direct) if isfile(join(direct, f))]
        sel = [s for s in files if "tfrecords" in s]
        sel = [direct + s for s in sel]
        selection_array = np.array(sel)
        filenames = np.append(filenames, selection_array)
    
    # get number of files
    n_files = filenames.shape[0]
    
    # get number of evaluation steps
    n_steps = np.int32(np.ceil(n_files/files_at_once))
    
    # loop over evaluation steps
    for i in range(n_steps):
        
        # take files
        files_test = filenames[(i*files_at_once):((i+1)*files_at_once)]
            
        # initiate model
        model = hourglass()

        # load weights
        model.load_weights('models/' + model_name + '.keras')

        # load dataset: get heatmap for all, the associated heatmap if existing (not for lakes/rivers) and the mercator coordinates
        if ground_truth != 'none':
            dataset_test = read_dataset(files_test, ground_truth = ground_truth, shuffle=False, epochs = 1)
        dataset_test_all = read_dataset(files_test, ground_truth = 'all', shuffle=False, epochs = 1)
        dataset_test_mercator = read_dataset(files_test, ground_truth ='mercator', shuffle=False, epochs = 1)

        # create a counter
        counter = 0

        # array to store information to combine predictions with; contains the locations of the images to later be able to calculate the predicted locations in mercator coordinates
        mercator = np.empty((1000000, 3))
        mercator[:] = np.NaN

        # loop over dataset
        for j, x in enumerate(dataset_test_mercator):
        
            # store mercator corner of image and the id
            mercator[counter:counter + x[0].shape[0], 0] = x[0].shape[0] * [j]
            mercator[counter:counter + x[0].shape[0], 1] = x[0][:, 0]
            mercator[counter:counter + x[0].shape[0], 2] = x[1][:, 0]
        
            # add to counter
            counter += x[0].shape[0]

        # remove empty rows
        mercator = mercator[~np.isnan(mercator).any(axis=1)]
    
        # get predictions
        predictions = model.predict(dataset_test_all)

        # get associated heatmaps and combine with np (not for rivers/lakes)
        if ground_truth != 'none':
            truth_specific = np.concatenate([true for x, true in dataset_test], axis=0)
            
        # get all dam heatmap
        truth_all = np.concatenate([true for x, true in dataset_test_all], axis=0)

        # save mercator coordinates
        np.save('output/' + output_path + '_' + model_name + '_' + str(i) + '_mercator.npy', mercator)

        # save all dams heatmap
        np.save('output/' + output_path + '_' + model_name + '_' + str(i) + '_all.npy', truth_all)

        # save associated dam heatmap
        if ground_truth != 'none':
            np.save('output/' + output_path + '_' + model_name + '_' + str(i) + '_' + ground_truth + '.npy', truth_specific)
        
        # save the predictions
        np.save('output/' + output_path + '_' + model_name + '_' + str(i) + '_pred.npy', predictions)



### main code

try:
    os.mkdir(directory + output_directory)
    os.mkdir(directory + model_directory)
except:
    print('Directory already exists')


# train model
train_model(training_directories = data_directory_training, negative_training_directories = negative_directory_training, ground_truth = g_truth, epochs = num_epochs)

# apply model
apply_model(testing_directories = [data_directory_testing[0]], output_path = output_short[0], ground_truth = g_truth_application[0])
apply_model(testing_directories = [data_directory_testing[1]], output_path = output_short[1], ground_truth = g_truth_application[1])
apply_model(testing_directories = [data_directory_testing[2]], output_path = output_short[2], ground_truth = g_truth_application[2])
apply_model(testing_directories = [data_directory_testing[3]], output_path = output_short[3], ground_truth = g_truth_application[3])
apply_model(testing_directories = [data_directory_testing[4]], output_path = output_short[4], ground_truth = g_truth_application[4])
apply_model(testing_directories = [data_directory_testing[5]], output_path = output_short[5], ground_truth = g_truth_application[5])
