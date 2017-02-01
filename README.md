# Behavioral Cloning

The purpose of the model is to be able to pass first track in the Udacity simulator. Simulator uses train mode to record training data, images, steering and other parameters, and autonomous mode for using trained model to produce steering in response to simulator images.

## Image Preprocessing

Udacity provided data is used as a starting point for training the model. The main drawback of provided data is it being unbalanced with bias towards going straight. Provided model addresses this in multiple ways. We take all center camera images with steering untouched and all left and right camera images altered by 0.25 and -0.25 respectively to adjust to cameras shift. Then we try to make the data more balanced by introducing more examples with larger steering. For non-zero steering we apply steering augmentation by 0.1, 0.15 or -0.1,-0.15 depending on angle sign for the center camera, 0.30, 0.35 for the left camera and -0.30, -0.35 for the right camera. Then all augmented data is appended to our training set. 

<p align="center">
  <img src="data/IMG/left_2016_12_01_13_30_48_287.jpg" width="30%"/>
  <img src="data/IMG/center_2016_12_01_13_30_48_287.jpg" width="30%"/>
  <img src="data/IMG/right_2016_12_01_13_30_48_287.jpg" width="30%"/>
</p>

Training set is shuffled before and after validation split and reshuffled in every epoch.

Validation and training set histograms after adding Augmentation and reshuffle:

<p align="center">
  <img src="hists/hist_validation.jpg" width="45%"/>
  <img src="hists/hist_training.jpg" width="45%"/>
</p>

## Model Fitting

Fit_generator function from Keras library is used to fit the model in memory efficient way: it loads images batch by batch in parallel with the model training.

The data is normalised in the model using Lambda function before the first convolution.

## Network Architecture

Network architecture is based on the network from NVIDIA article http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf 

It starts with a set of convolutional layers to introduce non-linearity to the model. First, there are three 5 by 5 convolutions with stride 2; first layer has valid padding and latter two layers have same padding. The depth of convolutions are 24, 36 and 48 respectively.

Then it is followed by two convolutional layers 3 by 3, depth 64, same padding.

Every convolution is activated with RELU.

Then network continues with 3 fully connected layers with 100, 50 and 10 neurons respectively. They are followed by 0.5 dropout layer to avoid overfitting.

Output layer with single neuron produces single output for predicted steering.

The model is using Adam optimiser (improved version of Gradient Descent) to minimise minimum square error loss function.




