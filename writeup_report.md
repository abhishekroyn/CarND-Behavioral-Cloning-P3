# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/data_distribution_before.png "Initial Data Distribution"
[image2]: ./output_images/data_distribution_after.png "Fitered Data Distribution"
[image3]: ./output_images/train_test_data_dist.png "Training Validation Data Distribution"
[image4]: ./output_images/compare_aug_zoom_org_img.png "Compare - Augmented zoomed Image vs Original Image"
[image5]: ./output_images/compare_aug_pan_org_img.png "Compare - Augmented Pan Image vs Original Image"
[image6]: ./output_images/compare_aug_bright_org_img.png "Compare - Augmented Brightness Image vs Original Image"
[image7]: ./output_images/compare_aug_flip_org_img.png "Compare - Augmented Filtered Image vs Original Image"
[image8]: ./output_images/compare_batch_aug_org_img.png "Compare - Augmented Batch Image vs Original Image"
[image9]: ./output_images/train_test_loss.png "Training Validation Loss Plot"
[image10]: ./output_images/train_test_acc.png "Training Validation Accuracy Plot"
[video1]: ./output_images/run1.mp4  "Output video with autonomous driving"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 capturing the output as driving the car in autonomous mode

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. The code in model.py uses a Python generator `fit_generator`, to preprocess data for training rather than preprocessing and storing all the training data in memory. The model.py code is clearly organized and comments are included wherever needed.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with
    - 5x5 filter sizes and depths between 24 and 48 (model.py lines 142-144)
    - 3x3 filter sizes and depth as 64 (model.py lines 145-146)
    - a flatten layer (model.py line 147)
    - dense layers number of neurons ranging between 100 and 1 (model.py lines 148-151)
    
The model includes ELU layers to introduce nonlinearity (code line 142-150), and the data is normalized in the model during preprocessing step by dividing the data value with max value, 255 (code line 113). 

#### 2. Attempts to reduce overfitting in the model

The model was experimented with dropout layers at various stages, especially one at the end of last `Convolution2D` layer and also after every `Dense` layer, with values varying between 0.5 and 0.75 but that didn't help as much to reduce overfitting as much as adding extra `Convolution2D` layers did at the beginning and adding extra `Dense` layers at the end of the model (model.py lines 142-151). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 195). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. Initial learning rate was initially chose as 0.001 and finally defined as 0.0001 for higher accuracy of the model (model.py line 153).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I covered 4 loops of training data while driving in both the direction on the track to ensure steering angles are not biased towards turning only one side in the training data.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to collect training data from simulated driving, preprocess collected input images, train a convolutional neural network based on the collected data, and thus predict expected steering angle given automated driving environment.

My first step was to use a convolution neural network model similar to the lenet model. I thought this model might be appropriate because this was a simple model with just two `Convolution2D` layers and three `Dense` layers.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. I realized while testing the car was always going off-road as soon as driving begins. It was overfitting.

To combat the overfitting, I modified the model and switched to another simpler model - nvidia model, so that the  the validation errors could be lowered. Adding more `Convolution2D` layers and `Dense` layers helped to overcome overfitting. I called this model `steeringNet_model`. I believe this name is not unique but it definitely looks like fits perfectly as per the use case here.

Also, I collected data while driving on track in both the directions to avoid steering angle and bias.

A large volume of data had steering angle as 0.0 due to the nature of the driving which was casusing additional bias toward 0.0 steering angle in the results. Thus, I filtered out data inputs for any specific angle while restricting maximum count of data allowed for specific angle to be 400 (lines 34-49).

I augmented my training and validation images.

Further, I preprocessed them as below (108-114),
    - cropped out part of the image from top and bottom in order to exclude sky and car dashboard regions from the images (line 109)
    - converted rgb images to yuov images and processed them using `cv2.GaussianBlur` (line 110-111)
    - resized images from (160, 320, 3) to (66, 200, 3) for aligning their size to be used with nvidia model (line 112)
    - normalized the data (line 113)

Then I played with few optimzers and zeroed it on to Adam optimizer as that includes momentum as well and tuning of learning rate was not needed manually.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I collected additional data with recordings only when vehicle is recovering from being at extreme ends of the lane lines and moving towards center of the lane-lines or track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 140-156) consisted of a convolution neural network with the following layers and layer sizes.

Here is a tabular-visualization of the architecture, output shape and paramteres count in each of the layers,

Initial image shape      : (160, 320, 3)
Preprocessed image shape : (66, 200, 3)
The preprocessed image is then fed to the model.

| Layer         		|     Description	        				                                      	         | 
|:---------------------:|:------------------------------------------------------------------------------------------:| 
| Input         		| 66x200x3 YUV image   							                                             | 
| Convolution2D     	| 24 filters, 5x5 kernel_size, 2x2 subsample, ELU activation, outputs 31x98x24, 1824 param   |
| Convolution2D     	| 36 filters, 5x5 kernel_size, 2x2 subsample, ELU activation, outputs 14x47x36, 21636 param  |
| Convolution2D     	| 48 filters, 5x5 kernel_size, 2x2 subsample, ELU activation, outputs 5x22x48, 43248 param   |
| Convolution2D     	| 64 filters, 3x3 kernel_size, 1x1 subsample, ELU activation, outputs 3x20x64, 27712 param   |
| Convolution2D     	| 64 filters, 3x3 kernel_size, 1x1 subsample, ELU activation, outputs 1x18x64, 36928 param   |
| Flatten       	    | outputs 1152, 0 param     									                             |
| Fully connected		| ELU activation, outputs 100, 115300 param                    							     |
| Fully connected		| ELU activation, outputs 50, 5050 param                    							     |
| Fully connected		| ELU activation, outputs 10, 510 param                    							         |
| Fully connected		| ELU activation, outputs 1, 11 param                    							         |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track using driving in one direction.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover towards center of the lane if in case it is going outside the track.

Then I repeated this process on the same track but in opposite direction than before in order to get more data points and also unbiased steering angle data as well.

After the collection process, I had 10,796 number of data points. Here is an image with initial data distribution as per their steering angle values:
![alt text][image1]

I then preprocessed this data by keeping only maximum 400 data points for any specific steering angle value, and discarding others. So, I removed 6,399 data points. Remaining data points left for processing was 4,397. Here is an image with filtered data distribution as per their steering angle values:
![alt text][image2]

As each of these data points had 3 additional data points for left, center and right camera angles, thus total number of available data points were 13,191.

I finally randomly shuffled the data set and put 20% of the data into a validation set. Thus, I had 10,552 training samples and 2,639 validation samples to work with. Here is an image with filtered training and validation data distribution as per their steering angle values:
![alt text][image3]

To augment the data sat I followed four primary transformations as mentioned below (lines 70-105),
    - zoom using `iaa.Affine(scale=(1, 1.3))`
    - pan using `iaa.Affine(translate_percent= {"x" : (-0.1, 0.1), "y": (-0.1, 0.1)})`
    - change brightness using `iaa.Multiply((0.2, 1.2))`
    - random flip using `cv2.flip(image, 1)` and `steering_angle = -steering_angle`

For example, here is an image that has then been zoomed:
![alt text][image4]
For example, here is an image that has then been panned:
![alt text][image5]
For example, here is an image that has then been changed brightness:
![alt text][image6]
For example, here is an image that has then been flipped:
![alt text][image7]

Further, I preprocessed them as below (108-114),
    - cropped out part of the image from top and bottom in order to exclude sky and car dashboard regions from the images (line 109)
    - converted rgb images to yuov images and processed them using `cv2.GaussianBlur` (line 110-111)
    - resized images from (160, 320, 3) to (66, 200, 3) for aligning their size to be used with nvidia model (line 112)
    - normalized the data (line 113)

I used this training data for training the model using batch data generator (line 195). For example, here is an image of batch augmented data as compared to original image data:
![alt text][image8]

The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the `loss` and `accuracy` plots below. I used an adam optimizer so that manually training the learning rate wasn't necessary.

For example, here is a plot with `loss` for training and validation data:
![alt text][image9]
For example, here is a plot with `accuracy` for training and validation data:
![alt text][image10]

### Simulation

#### 1. Is the car able to navigate correctly on test data?

The car was driven autonomously and correctly around the track one.
`video.mp4` captured the output as driving the car in autonomous mode. Here's a [link to my video result - video.mp4](./output_images/run1.mp4)

