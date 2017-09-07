# Deep learning projects

## Overview 

This repository contains some of projects I achieved  in order to learn about neural networks. It focus on deep-learning, using different types of neural networks and approaches .

## Project 1:

The project 1 focus on building a Neural Network to predict the number of bikeshare users on a given day. Imagine yourself owning a bikesharing company. You want to predict how many bycles you need because if they are too few, you will loose money from potential rides, if they are too many, you will waste money on buying cycles just settled around.  You need to predict from historical datas how money bycles you will need to buy in the future.
![bikesharing](https://github.com/virt87/DeepLearning/blob/master/Bikeshare-neural-network/bikesharing.png)

### Dependencies

* Download anaconda
* Create a new conda environment:
```
conda create --name bikesharing python=3
```
* Activate the source
```
source activate bikesharing
```
* Ensure you have numpy, matplotlib, pandas, and jupyter notebook installed by doing the following:
```
conda install numpy matplotlib pandas jupyter notebook
```
* Run the following to open up the notebook:
```
jupyter notebook bikesharing-neural-network.ipynb
```
### Take a look
You can open the jupyter notebook directly in github [bikesharing-neural-network.ipynb](https://github.com/virt87/DeepLearning/blob/master/Bikeshare-neural-network/bikesharing-neural-network.ipynb)

## Project 2:
Sentiment analysis with Andrew Trask. The main focus of this project is to understand how to create a neural network that can anlyse a review and classify it as bad or good.
It is split into several parts:
- curating a data set.
- training a a neural network
- increasinghe signal and reducing the noise in teh data set.

### Dependencies

* Download anaconda
```
conda create --name sentiment_alysis python=3
```

* Activate the source
```
source activate sentiment_alysis
```
* Ensure you have numpy, jupyter notebook, matplotlib, scikit-learn, and bokeh installed by doing the following:
```
conda install numpy matplotlib  jupyter notebook scikit-learn bokeh
```
* Run the following to open up the notebook:
```
jupyter notebook Sentiment\ Classification.ipynb
```


### Take a look
You can open the jupyter notebook directly in github [Sentiment Classification.ipynb](https://github.com/virt87/DeepLearning/blob/master/Sentiment analysis/Sentiment Classification.ipynb)


## Project 3:
Sentiment analysis with TFLearn. The same work as the previous project by now using TFLearn.

### Dependencies

* Download anaconda
```
conda create -n tflearn python=3.5
```

* Activate the source
```
source activate tflearn
```
* Ensure you have numpy, jupyter notebook, matplotlib and pandas.
```
conda install numpy pandas jupyter notebook matplotlib
```
* Run the following to open up the notebook:
```
jupyter notebook Sentiment\ Analysis\ with\ TFLearn.ipynb
```

### Take a look
You can open the jupyter notebook directly in github [Sentiment Analysis with TFLearn.ipynb](https://github.com/virt87/DeepLearning/blob/master/TFLearn/sentiment-analysis/Sentiment Analysis with TFLearn.ipynb)



## Project 4:
Handwritten digit recognition with TFLearn. Using the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset, which contains images of handwitten single digits and their respective lavels (numbers from 0 to 9), we train a neuronal network that recognizes handwritten digits.

### Dependencies

* Download anaconda
```
conda create -n tflearn python=3.5
```

* Activate the source
```
source activate tflearn
```
* Ensure you have numpy, jupyter notebook, matplotlib and pandas.
```
conda install numpy pandas jupyter notebook matplotlib
```
* Run the following to open up the notebook:
```
jupyter notebook Handwritten\ Digit\ Recognition\ with\ TFLearn.ipynb
```

### Take a look
You can open the jupyter notebook directly in github [Handwritten Digit Recognition with TFLearn.ipynb](https://github.com/virt87/DeepLearning/blob/master/TFLearn/Handwritten%20Digit%20Recognition/Handwritten%20Digit%20Recognition%20with%20TFLearn.ipynb)

## Project 5:
In this project, I'll classify images from the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. The dataset consists of airplanes, dogs, cats, and other objects. The dataset will need to be preprocessed, then train a convolutional neural network on all the samples. I'll normalize the images, one-hot encode the labels, build a convolutional layer, max pool layer, and fully connected layer. At then end, I'll see their predictions on the sample images.

As an exercise, this project will run on a machine with GPU using floydhub.com.

### Take a look
You can open the jupyter notebook directly in github [Image Classification](https://github.com/sbatururimi/DeepLearning/blob/master/Image%20Classification/dlnd_image_classification.ipynb)


## Project 6:
In this project, I played with a Recurren Neural network implementing LSTM cells in order to generate my own Simpsons TV script. I used a part of the Simpsons dataset of scripts from 27 seasons. The Neural Network I built  generate a new TV script for a scene at Moe's Tavern. The RNN was trained on GPU using FloydHub.

In order to run in, setup Conda, then activate an environment:
```
$ conda env create -f environment.yaml
```
Setup floyd hub and launch it in GPU mode:
```
$ floyd login
$ floyd init
$ floyd run --mode jupyter --gpu
```

### Take a look
You can open the jupyter notebook directly in github 
 [TV script generation with RNN](https://github.com/sbatururimi/DeepLearning/blob/master/tv-script-generation/dlnd_tv_script_generation.ipynb)

