#**Behavioral Cloning** 

##Writeup Report


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/center.jpg "Center Image"
[image3]: ./examples/recovery1jpg "Recovery Image"
[image4]: ./examples/recovery2.jpg "Recovery Image"
[image5]: ./examples/recovery3.jpg "Recovery Image"
[image6]: ./examples/normal.jpg "Normal Image"
[image7]: ./examples/flipped.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes for depths sizes 24, 36 and 48 and 3x3 filter sizes for two conolution layes with depth size 64. (model.py lines 69-80) 

The model includes RELU layers to introduce nonlinearity for almost every layer, and the data is normalized in the model using a Keras lambda layer (code line 65). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 74 and 85). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 92).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and couple of laps of reverse driving so that the model generalizes better.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was based on the Nvidea self driving car research paper provided.

My first step was to use a convolution neural network model similar to the Nvidea research paper. I thought this model might be appropriate because it was a proven technique.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to add Dropout layers.

Then I tuned the Dropout parameter so as to reduuce overfitting and decrease mse value. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track like around and on bridges and a sharp turn, to improve the driving behavior in these cases, I added other two camera images to the data, flipped the original data and added additional data for places on the track where the car was having trouble staying on the track (bridge and very sharp curves)

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 69-91) consisted of a convolution neural network with the following layers and layer sizes
5x5 Convolution
2x2 Subsampling
Relu Activation
5x5 Convolution
2x2 Subsampling
Relu Activation
Droput
5x5 Convolution
2x2 Subsampling
Relu Activation
3x3 Convolution
Relu Activation
3x3 Convolution
Relu Activation
100 Fully connected
Relu Activation
Dropout
50 Fully connected
Relu Activation
1 Fully connected

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from the side of the road. These images show what a recovery looks like starting from the side of the road to the center. :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would generalize the model as most of the turns in the tracks are to the left so the model won't be left leaning. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


After the collection process, I had 21204 number of data points. I then preprocessed this data by normalizing it using lambda layer and then cropping it.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by no decrease in mse after 5 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
