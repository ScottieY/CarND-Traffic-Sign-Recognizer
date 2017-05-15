# **Traffic Sign Recognition** 

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is  32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data are spreaded

![download](https://github.com/ScottieY/CarND-Traffic-Sign-Recognizer/blob/master/download.png)

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because this reduce the effect of color which varies with light and focus on the shape of the sign.

Here is an example of a traffic sign image before and after grayscaling.

![rgb](https://github.com/ScottieY/CarND-Traffic-Sign-Recognizer/blob/master/rgb.png)

![gray](https://github.com/ScottieY/CarND-Traffic-Sign-Recognizer/blob/master/gray.png)

As a last step, I normalized the image data because normalize the input range from [0,255] to [-0.5,0.5] could make training faster, and avoid get stuck in local minimum. 

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The given data set was already splitted into train, valid and test set.

My final training set had 34799 number of images. My validation set and test set had 4410 and 12630 number of images.




#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

|      Layer      |               Description                |
| :-------------: | :--------------------------------------: |
|      Input      |            32x32x1 Gray image            |
| Convolution 5x5 | 1x1 stride, same padding, outputs 32x32x6 |
|      RELU       |                                          |
|   Max pooling   |       2x2 stride,  outputs 16x16x6       |
| Convolution 5x5 | 1x1 stride, same padding, outputs 16x16x12 |
|      RELU       |                                          |
|   Max pooling   |       2x2 stride,  outputs 8x8x12        |
| Fully connected |               8x8x12 to 80               |
|      RELU       |                                          |
|    Drop out     |                   0.75                   |
| Fully connected |                 80 to 80                 |
|      RELU       |                                          |
|    Drop out     |                   0.75                   |
|     Output      |                 80 to 43                 |



#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an adam optimizer to minimize the cost.



| Hyper parameter | Value  |
| --------------- | ------ |
| Learning rate   | 0.0008 |
| EPOCHS          | 100    |
| BATCH_SIZE      | 128    |
| Drop out        | 0.5    |



#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the eighth and ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.943
* test set accuracy of 0.927


* What architecture was chosen?

  LeNet was chosen

* Why did you believe it would be relevant to the traffic sign application?

  It did well on the MNIST data recognition

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

  After tunning some of the hyper parameter and adding an extra layer. It gives really good accuracy stated above.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![t1](https://github.com/ScottieY/CarND-Traffic-Sign-Recognizer/blob/master/t1.png)

The road narrow on the right and roundabout mandatory sign occupied the entire image may be hard to classify, since the training images are typically smaller.

The stop sign boarder on the left and right is a little bit blurry.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

|           Image           |        Prediction         |
| :-----------------------: | :-----------------------: |
| Road narrows on the right | Road narrows on the right |
|           Stop            |           Stop            |
|           Yield           |           Yield           |
|          30 km/h          |          30 km/h          |
|   Roundabout mandatory    |   Roundabout mandatory    |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 92%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability |              Prediction               |
| :---------: | :-----------------------------------: |
|  0.000467   |          Beware of ice/snow           |
|   0.0026    | Right-of-way at the next intersection |
|   0.00385   |           Children crossing           |
|   0.0149    |              Pedestrians              |
|    0.978    |       Road narrows on the right       |

For the second image 

| Probability |      Prediction      |
| :---------: | :------------------: |
|  0.000039   | Go straight or left  |
|  0.000059   | Speed limit (30km/h) |
|  0.000288   |        Yield         |
|   9.9876    |         Stop         |
|  0.000822   |   Turn right ahead   |

For the third image

| Probability |      Prediction      |
| :---------: | :------------------: |
|      0      |      Ahead only      |
|      0      |      No passing      |
|      0      | Speed limit (50km/h) |
|      1      |        Yield         |
|      0      |     No vechiles      |

For the fourth image

| Probability |      Prediction      |
| :---------: | :------------------: |
|      0      |   General caution    |
|      0      | Speed limit (70km/h) |
|      1      | Speed limit (30km/h) |
|      0      | Speed limit (50km/h) |
|      0      | Speed limit (20km/h) |

For the fifth image

| Probability |                Prediction                |
| :---------: | :--------------------------------------: |
|      0      |          Speed limit (100km/h)           |
|      0      | End of no passing by vechiles over 3.5 metric tons |
|      1      |           Roundabout mandatory           |
|      0      | Vechiles over 3.5 metric tons prohibited |
|      0      |              Priority road               |
