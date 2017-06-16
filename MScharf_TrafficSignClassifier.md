#**Traffic Sign Recognition** 

This documents describes my approach to solve the Traffic Sign Recognition project according to the requirements. I've implemented a solution in python, not using the jupyter notebook template, in order to have more flexibility. During the development I've played around with different hyperparameters for the CNN as well as with the layout of the NN itself. Furthemrore I've made some investigations and adaptations on the test-data in order to realize a optimal learning input. 
The following document is starting with a short overview of the goals for the project as it was given from Udacity. After that a short introduction to the architecture of the python solution is given. The following sections gives some details about the training set we've used for testing. With these findings, I'll provide some details about the preprocessing of the training data I've applied in order to get a better training set. After this sections I'll describe the model I've choosen for the project and give some comparisms of other testruns I did. The final section will summarize the performance on testsamples I've taken from my surrounding.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/testsample_distribution.jpeg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###TrafficSignClassifier.py overview
The first questions I've asked myself - how to deal with the degree of freedom that is given in a CNN like the LeNet architecture. I was expecting to make a bunch of measurements, changing some parameters and rerun the test. At the end a compareable result must be summarized. In order to make my life easier, I've decided to create a solution that is not based on the structure given in the jupyter notebook template. This solution allows me to predefine different settings of parameters and run the tests single/ multithreaded in a automatic sequence e.g. over night. 

The TrafficSignClassifier.py contains 3 classes (the TrafficSignClassifier and two helpers for logging) and a main function which is used to start the reading/training/verification process.  I'll concentrate on the TrafficSignClassifier - further abbreviated as TSC - only - the LoggingClasses are used to write the results to console as well as in a textfile.

#####statics

TSC contains some static as well as instance members. The intention behind the static members and functions is, that reading training-data as well as applying different transformations on this data should be done once, and then shared among all the configurations.

As you can imaging: `X, Y train/valid/test` contains the testdata read from the file. The `classes` member contains the association between label 0-42 and the corresponding textual representation. The class is storing an overview of the samples for each label in the `trainingLabelSetIndex` - meaning for each label I've the index to a training sample. The class member `id` is used in the multithreaded environment to associate log-outputs with the corresponding configuration setup.

Let review in short the static class methods:
######importData(filepath):
Searches in the `filepath` for the files test.p/train.p/vlaid.p and signames.csv and uses `pickle` and `csv.DictReader` to read the data to the static classmembers mentioned above. 
######__printSummary():
Provides a textual output based on the classmemebrs as expected by the jupyter notebook template in form of
> Basic Summary of the DataSet
> 	Number of training examples =  34799
> 	Number of testing examples =  12630
> 	Image data shape =  (32, 32)
> 	Number of classes =  43

Please note - this member is intented to be private and will be get called automatically e.g. in `importData`

######preAnalyzeData()
This method provides a graphical representation of the samples for each label - the figure below shows you the exact distribution of label - samples.
>**TODO**

![alt text][image1]

######__preAnalyzeData(force = False)
Again a private method which is getting called from other functions. In essence it uses the input data to fill the member `trainingLabelSetIndex` based on the given data. If the function was called once, it will immediately return in case that is is getting called another time. Only by forcing the function to reinitialize, the `trainingLabelSetIndex` will be updated. This method is used in order to visualize the augmented data set after aligning the samples for each label


######simpleDataAugmentation()
The first try on augmentation was to simply adapt the number of samples for each label. This was done by identifying the label with the most samples and then cloning for each other label the samples so that at the end each label has the same number of test samples. During my testing I've figured out that beneath the drastically increased runtime, the performance of the CNN wasn't improving that good - so I decided to realize another function as described in the next section. 

######dataAugmentation(size = None)
As mentioned - the number of testsamples isn't equal for each label. Therefore I've realized a method which will change the samples in the following way. The paremeter `size` is used to specify the number of each sample for each label - `None` means to set the sample count to the max sample count of the given testdata - in our case **TODO**.
So what is happening in this function. For each lable I choose randomly `size` sample and then:
* use the first sample as it is
* use the second sample and rotate it for +15° without changing shape
* use the third sample and shift it along x,y axis for 2 pixel without chaning shape
* apply the operations on the next three samples

######drawDataSetExample()
Take for each lable one sample and draw it via matplot.


#####instance members
A TSC instance is operating on the static members described above. The instance members reflecting configurations and  operations as well as the layout of the underlying CNN network architecture. Each of the TSC instances is making a personal copy of the training/ validation/ test set so that any changes on the data is not interfering with other instances of the TSC running. I'll not go into the details of the member variables - these are more or less self-explaining. Beneath the testset the member variables represent the tensors used for training and validation.

Let discuss the member functions

######normalize_zeroMeanData(self)
As a suggestion from the jupyter notebook template, i've implemented the zeroMean data normalization. This is done by changing each pixel using `X = (X-128) / 128` which will result in a mean of zero and a deviation of 1. I've noticed that this step improves the matching rate of the CNN immense - so in principle this is a standard operation i've applied to each testrun.

######convertToGrayScale_luminosity(self)
Another suggestion from the template was to convert the images/ samples into grayscaled image. This is happening in this member function. For each pixel I'm applying `X_gray = X_red * 0.21 + X_green * 0.72 + X_blue * 0.07`. 
The results/ effects of this transformations were negligible during my testruns - so I'm not applying this operation at all.

######__LeNet(self,cfg, x)
As the name said, I'm staying with the LeNet architecture for my NN. The major difference to the LeNet architecture is the ability to configure via the `cfg` paraemter
* mu & sigma for the randomly created weights/ bias
* the scale of filters/ strides
* the scale of filters/ strides for max pooling
* the depth in the three fully connnected layers
* and an optional third convolution layer

Furthermore I've added a configureable **dropout** layer between FullyConnected1 and FullyConnected2 to prevent overfitting.

The remaining setup is more or less unchanged - meaning 
* relu activation functions
* truncated_normal random functions for weights/ biases
* valid padding for filter-applying

And for sure - due to the fact that our resulting space contains 43 labels, the number of resulting logits is changed from 10 to 43.

**Please note**: this function is intended to be private - so it should only be called from other member functions, e.g. TrainCNN.

######TrainCNN(self,cfg):
This is more or less the entry point for each TSC object. During TrainCNN a tensorFlow session is startet. The sample-data is feeded via placeholders to the learning pipeline. As in the classical LeNet architecture our loss is measured as cross entropy of our results and the labels as OneHot encoded. This is translated into propabilities by using the softmax function. The resulting loss is given then by the reduced mean of these cross-entropy propabilities over the whole testsamples. The optimization - using AdamOptimizer - is running EPOCHS number of times. 
As before - the `cfg` parameter is used to configure 
* Batchsize which is by default 128
* number of epochs which is by default 10
* the learning rate which is by defaul 0.001

In case that our validation accuraty reaches the 0.93 mark after training, we immediately apply the NN to the training data. This is done via the `__EvaluateCNN` member.

######__EvaluateCNN(self, cfg, X_data, y_data):
Evaluation of the trained NN using the test data. In contrast to training we've associated the dropout keep propability to 1.0. The accuracy is given by comparing the NN result for a sample with the label for the sample - which is either true (1) or false (0). The mean of this comparision over all samples provides the overall accuracy.

#####the confgiuration



###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
* The size of the validation set is ?
* The size of test set is ?
* The shape of a traffic sign image is ?
* The number of unique classes/labels in the data set is ?

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

