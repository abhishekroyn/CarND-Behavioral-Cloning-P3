import os
import csv
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from imgaug import augmenters as iaa
from keras import backend as K

# load data from csv
def load_data_from_csv(datadir, csvfilename):
    lines = []
    with open(os.path.join(datadir, csvfilename)) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

# distribute dataset into bins
def distribute_dataset(data, num_bins = 25):
    hist, bins = np.histogram(data['steering'], num_bins)
    return hist, bins
     
# filter dataset
def filter_dataset(data, bins, samples_per_bin = 400):
    print('total data:', len(data))
    remove_list = []
    for j in range(num_bins):
        list_ = []
        for i in range(len(data['steering'])):
            if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
                list_.append(i)
        list_ = shuffle(list_)
        list_ = list_[samples_per_bin:]
        remove_list.extend(list_)
     
    print('removed:', len(remove_list))
    data.drop(data.index[remove_list], inplace=True)
    print('remaining:', len(data))
    return data

# list data from all three cameras
def load_imagepath_steering(datapath, data):
    image_path = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        steering_center = float(indexed_data[3])
        for i in range(3):
            source_path = indexed_data[i]
            filename = source_path.split('/')[-1]
            image_path.append(os.path.join(datapath, filename))
        steering.append(steering_center)
        steering.append(steering_center + steering_correction)      # steering left
        steering.append(steering_center - steering_correction)      # steering right
    image_paths = np.asarray(image_path)
    steerings = np.asarray(steering)
    return image_paths, steerings

# image augmentation - zoom
def zoom(image):
  zoom = iaa.Affine(scale=(1, 1.3))
  image = zoom.augment_image(image)
  return image

# image augmentation - pan
def pan(image):
  pan = iaa.Affine(translate_percent= {"x" : (-0.1, 0.1), "y": (-0.1, 0.1)})
  image = pan.augment_image(image)
  return image

# image augmentation - brightness
def img_random_brightness(image):
    brightness = iaa.Multiply((0.2, 1.2))
    image = brightness.augment_image(image)
    return image

# image augmentation - flip
def img_random_flip(image, steering_angle):
    image = cv2.flip(image, 1)
    steering_angle = -steering_angle
    return image, steering_angle

# image augmentation - random
def random_augment(image, steering_angle):
    image = mpimg.imread(image)
    if np.random.rand() < 0.5:
      image = pan(image)
    if np.random.rand() < 0.5:
      image = zoom(image)
    if np.random.rand() < 0.5:
      image = img_random_brightness(image)
    if np.random.rand() < 0.5:
      image, steering_angle = img_random_flip(image, steering_angle)
    
    return image, steering_angle

# image preprocessing
def img_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img

# batch generation of augmented images
def batch_generator(image_paths, steering_ang, batch_size, istraining):
    while True:
        batch_img = []
        batch_steering = []
    
        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)
      
            if istraining:
                im, steering = random_augment(image_paths[random_index], steering_ang[random_index])
     
            else:
                im = mpimg.imread(image_paths[random_index])
                steering = steering_ang[random_index]
      
            # print(im.shape)
            im = img_preprocess(im)
            # print(im.shape)
            batch_img.append(im)
            batch_steering.append(steering)
        yield (np.asarray(batch_img), np.asarray(batch_steering))  

# steeringNet model created from base model as nvidia model
def steeringNet_model():
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='elu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Flatten())
    model.add(Dense(100, activation = 'elu'))
    model.add(Dense(50, activation = 'elu'))
    model.add(Dense(10, activation = 'elu'))
    model.add(Dense(1))

    optimizer = Adam(lr=1e-4)
    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

    return model

# user defined variables and configurations
datadir = 'examples/Track'
csvfilename = 'driving_log.csv'
datasubdir = 'IMG'
cols = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
steering_correction = 0.2
num_bins = 25
samples_per_bin = 400

# load data from csv
lines = load_data_from_csv(datadir, csvfilename)

# extract steering data from dataset
data = pd.DataFrame(lines, columns = cols)                  # list to dataframe
data['steering'] = data['steering'].astype(np.float)

# distribute dataset into bins
hist, bins = distribute_dataset(data, num_bins)

# filter dataset
data = filter_dataset(data, bins, samples_per_bin)

# distribute dataset into bins
hist, bins = distribute_dataset(data, num_bins)

# list data from all three cameras 
image_paths, steerings = load_imagepath_steering(os.path.join(datadir, datasubdir), data)

# split data into training and testing
X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size = 0.2, random_state = 6)
print('Training Samples: {}\nValid Samples: {}'.format(len(X_train), len(X_valid)))

# load steeringNet model
model = steeringNet_model()
print(model.summary())

# training and validation testing
history = model.fit_generator(batch_generator(X_train, y_train, 100, 1),
                                  steps_per_epoch = 300, 
                                  epochs = 10,
                                  validation_data = batch_generator(X_valid, y_valid, 100, 0),
                                  validation_steps = 200,
                                  verbose = 1,
                                  shuffle = 1)

# save model
model.save('model.h5')

# clear session
K.clear_session()
