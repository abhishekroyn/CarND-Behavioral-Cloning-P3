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

# DEBUGGING BEGINS
# ALTERNATE - extract steering data from dataset
if False:
    lines_array = np.asarray(lines)                         # list to array
    steering_array = lines_array[:, 3].astype(np.float)     # string to float
    steering = pd.Series(steering_array.tolist())           # array to list to series
# DEBUGGING ENDS

# display sample data
def display_sample_data(data):
    pd.set_option('display.max_colwidth', -1)
    print(data.head())

# distribute dataset into bins
def distribute_dataset(data, num_bins = 25):
    hist, bins = np.histogram(data['steering'], num_bins)
    return hist, bins

# DEBUGGING BEGINS
# plot data distriution
def plot_data_distribution(hist, bins, data, samples_per_bin = 400):
    center = (bins[:-1]+ bins[1:]) * 0.5
    plt.bar(center, hist, width=0.05)
    plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
    plt.savefig('output_images/data_distribution.jpg')
    plt.show()
# DEBUGGING ENDS
     
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

# DEBUGGING BEGINS
# plot trainging and testing data distribution
def plot_train_test_data_dist(y_train, y_valid, num_bins):
    fig, axes = plt.subplots(1, 2, figsize = (12, 4))
    axes[0].hist(y_train, bins = num_bins, width = 0.05, color = 'blue')
    axes[0].set_title('Training set')
    axes[1].hist(y_valid, bins = num_bins, width = 0.05, color = 'red')
    axes[1].set_title('Validation set')
    plt.savefig('output_images/train_test_data_dist.jpg')
    plt.show()
# DEBUGGING ENDS

# image augmentation - zoom
def zoom(image):
  zoom = iaa.Affine(scale=(1, 1.3))
  image = zoom.augment_image(image)
  return image

# DEBUGGING BEGINS
# display augmented zoom image as compared to original image
def plot_compare_aug_zoom_org_img(image_paths):
    image = image_paths[random.randint(0, 1000)]
    original_image = mpimg.imread(image)
    zoomed_image = zoom(original_image)
     
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    fig.tight_layout()
     
    axs[0].imshow(original_image)
    axs[0].set_title('Original Image')
     
    axs[1].imshow(zoomed_image)
    axs[1].set_title('Zoomed Image')

    plt.savefig('output_images/compare_aug_zoom_org_img.jpg')
    plt.show()
# DEBUGGING ENDS

# image augmentation - pan
def pan(image):
  pan = iaa.Affine(translate_percent= {"x" : (-0.1, 0.1), "y": (-0.1, 0.1)})
  image = pan.augment_image(image)
  return image

# DEBUGGING BEGINS
# display augmented pan image as compared to original image
def plot_compare_aug_pan_org_img(image_paths):
    image = image_paths[random.randint(0, 1000)]
    original_image = mpimg.imread(image)
    panned_image = pan(original_image)
     
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    fig.tight_layout()
     
    axs[0].imshow(original_image)
    axs[0].set_title('Original Image')
     
    axs[1].imshow(panned_image)
    axs[1].set_title('Panned Image')

    plt.savefig('output_images/compare_aug_pan_org_img.jpg')
    plt.show()
# DEBUGGING ENDS

# image augmentation - brightness
def img_random_brightness(image):
    brightness = iaa.Multiply((0.2, 1.2))
    image = brightness.augment_image(image)
    return image

# DEBUGGING BEGINS
# display augmented brightness image as compared to original image
def plot_compare_aug_bright_org_img(image_paths):
    image = image_paths[random.randint(0, 1000)]
    original_image = mpimg.imread(image)
    brightness_altered_image = img_random_brightness(original_image)
     
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    fig.tight_layout()
     
    axs[0].imshow(original_image)
    axs[0].set_title('Original Image')
     
    axs[1].imshow(brightness_altered_image)
    axs[1].set_title('Brightness altered image ')

    plt.savefig('output_images/compare_aug_bright_org_img.jpg')
    plt.show()
# DEBUGGING ENDS

# image augmentation - flip
def img_random_flip(image, steering_angle):
    image = cv2.flip(image, 1)
    steering_angle = -steering_angle
    return image, steering_angle

# DEBUGGING BEGINS
# display augmented flip image as compared to original image
def plot_compare_aug_flip_org_img(image_paths, steerings):
    random_index = random.randint(0, 1000)
    image = image_paths[random_index]
    steering_angle = steerings[random_index]
     
    original_image = mpimg.imread(image)
    flipped_image, flipped_steering_angle = img_random_flip(original_image, steering_angle)
     
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    fig.tight_layout()
     
    axs[0].imshow(original_image)
    axs[0].set_title('Original Image - ' + 'Steering Angle:' + str(steering_angle))
     
    axs[1].imshow(flipped_image)
    axs[1].set_title('Flipped Image - ' + 'Steering Angle:' + str(flipped_steering_angle))

    plt.savefig('output_images/compare_aug_flip_org_img.jpg')
    plt.show()
# DEBUGGING ENDS

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

# DEBUGGING BEGINS
# display batch generated augmented image as compared to original image
def plot_compare_batch_aug_org_img(X_train, y_train, X_valid, y_valid):
    x_train_gen, y_train_gen = next(batch_generator(X_train, y_train, 1, 1))
    x_valid_gen, y_valid_gen = next(batch_generator(X_valid, y_valid, 1, 0))
     
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    fig.tight_layout()
     
    axs[0].imshow(x_train_gen[0])
    axs[0].set_title('Training Image')
     
    axs[1].imshow(x_valid_gen[0])
    axs[1].set_title('Validation Image')

    plt.savefig('output_images/compare_batch_aug_org_img.jpg')
    plt.show()
# DEBUGGING ENDS

# steeringNet model created from base model as nvidia model
def steeringNet_model():
    model = Sequential()
#    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
#    model.add(Cropping2D(cropping = ((70, 25), (0, 0))))
#    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='elu'))
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
    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])          # model.compile(loss = 'mse', optimizer = 'adam')

    # DEBUGGING BEGINS
    # lenet model
    if False:
        model = Sequential()
        model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
        model.add(Cropping2D(cropping = ((70, 25), (0, 0))))         # pixels from ((top, bottom), (left, right))
        model.add(Convolution2D(6, 5, 5, activation = "relu"))
        model.add(MaxPooling2D())
        model.add(Convolution2D(6, 5, 5, activation = "relu"))
        model.add(MaxPooling2D())
        mode.add(Flatten())                 #mode.add(Flatten(input_shape = (160, 320, 3)))
        model.add(Dense(120))
        model.add(Dense(84))
        model.add(Dense(1))
    # DEBUGGING ENDS

    # DEBUGGING BEGINS
    # nvidia model
    if False:
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
    # DEBUGGING ENDS

    return model

# DEBUGGING BEGINS
# plot training and testing loss
def plot_train_test_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'validation'], loc='upper left')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('output_images/train_test_loss.jpg')
    plt.show()
# DEBUGGING ENDS

# DEBUGGING BEGINS
# plot training and testing accuracy
def plot_train_test_acc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.legend(['training', 'validation'], loc='upper left')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('output_images/train_test_acc.jpg')
    plt.show()
# DEBUGGING ENDS

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

# DEBUGGING BEGINS
if True:
    # display sample data
    display_sample_data(data)
# DEBUGGING ENDS

# distribute dataset into bins
hist, bins = distribute_dataset(data, num_bins)

# DEBUGGING BEGINS
if True:
    # plot data distriution
    plot_data_distribution(hist, bins, data, samples_per_bin)
# DEBUGGING ENDS

# filter dataset
data = filter_dataset(data, bins, samples_per_bin)

# distribute dataset into bins
hist, bins = distribute_dataset(data, num_bins)

# DEBUGGING BEGINS
if True:
    # plot data distriution
    plot_data_distribution(hist, bins, data, samples_per_bin)
# DEBUGGING ENDS

# list data from all three cameras 
image_paths, steerings = load_imagepath_steering(os.path.join(datadir, datasubdir), data)

# split data into training and testing
X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size = 0.2, random_state = 6)
print('Training Samples: {}\nValid Samples: {}'.format(len(X_train), len(X_valid)))

# DEBUGGING BEGINS
if True:
    # plot trainging and testing data distribution
    plot_train_test_data_dist(y_train, y_valid, num_bins)

    # display augmented zoom image as compared to original image
    plot_compare_aug_zoom_org_img(image_paths)

    # display augmented pan image as compared to original image
    plot_compare_aug_pan_org_img(image_paths)

    # display augmented brightness image as compared to original image
    plot_compare_aug_bright_org_img(image_paths)

    # display augmented flip image as compared to original image
    plot_compare_aug_flip_org_img(image_paths, steerings)

    # display batch generated augmented image as compared to original image
    plot_compare_batch_aug_org_img(X_train, y_train, X_valid, y_valid)
# DEBUGGING ENDS

# load steeringNet model
model = steeringNet_model()
print(model.summary())

# DEBUGGING BEGINS
# ALTERNATE - using model.fit without additional data generator
if False:
    model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 5)
# DEBUGGING ENDS

# training and validation testing
history = model.fit_generator(batch_generator(X_train, y_train, 100, 1),
                                  steps_per_epoch = 300, 
                                  epochs = 10,
                                  validation_data = batch_generator(X_valid, y_valid, 100, 0),
                                  validation_steps = 200,
                                  verbose = 1,
                                  shuffle = 1)

# DEBUGGING BEGINS
if True:
    # plot training and testing loss
    plot_train_test_loss(history)

    # plot training and testing accuracy
    plot_train_test_acc(history)
# DEBUGGING ENDS

# save model
model.save('model.h5')

# clear session
K.clear_session()
