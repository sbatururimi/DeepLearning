# Deep learning projects

## Overview 

This repository lists several deep learning project I achieved, using different types of neural networks and approaches.

## Project 1:
Predict the number of bikeshare users on a given day by building my own deep-learning library.
https://github.com/sbatururimi/bikeshare_neural_network

## Project 2:
Sentiment analysis with Andrew Trask, a project that classify a movie's review as positive or negative.
https://github.com/sbatururimi/sentiment_analysis

## Project 3:
Sentiment analysis with TFLearn. The same work as the previous project by now using TFLearn.
https://github.com/sbatururimi/sentiment_analysis_TFLearn


## Project 4:
Handwritten digit recognition with TFLearn. Using the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset, which contains images of handwitten single digits and their respective lavels (numbers from 0 to 9), we train a neuronal network that recognizes handwritten digits.
https://github.com/sbatururimi/Handwritten-Digit-Recognition-TFLearn.git


## Project 5:
Classify images from the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset using a Convolutional Network with Tensorflow. Trained on a machine with GPU in [floydhub](https://www.floydhub.com).

https://github.com/sbatururimi/image_classification_deep_learning

### Take a look
You can open the jupyter notebook directly in github [Image Classification](https://github.com/sbatururimi/DeepLearning/blob/master/Image%20Classification/dlnd_image_classification.ipynb)


## Project 6:
In this project, I played with a Recurrent Neural network implementing LSTM cells in order to generate my own Simpsons TV script. I used a part of the Simpsons dataset of scripts from 27 seasons. The Neural Network I built  generate a new TV script for a scene at Moe's Tavern. The RNN was trained on GPU using FloydHub.

In order to run it, setup Conda, then activate an environment:
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

## Project 7:
A case study of Transfer Learning.

In practice, you won't typically be training your own huge networks. There are multiple models out there that have been trained for weeks on huge datasets like ImageNet. In this project, I'll be using one of these pretrained networks, VGGNet, to classify images of flowers.

[Transfer Learning](https://github.com/sbatururimi/DeepLearning/blob/master/transfer-learning/Transfer_Learning.ipynb)

## Project 8:

A languag translation project using Sequence to Sequence model. It has been trained on  GPU, using Floyd.
This Sequence to Sequence model is trained on a dataset of English and French sentences and can translate new sentences from English to French.

[Language translation](https://github.com/sbatururimi/DeepLearning/blob/master/language-translation/dlnd_language_translation.ipynb)


## Project 9: 
Building an Autoencoder Neural network to compress and denoise images

[Compress without convolutions](https://github.com/sbatururimi/DeepLearning/blob/master/autoencoder/Simple_Autoencoder.ipynb)

[Compress and denoise with convolutions](https://github.com/sbatururimi/DeepLearning/blob/master/autoencoder/Convolutional_Autoencoder.ipynb)

## Project 10:
Training a GAN on MNIST to generate new handwritten digits

[A generative adversarial network (GAN) trained on the MNIST dataset.](https://github.com/sbatururimi/DeepLearning/blob/master/gan_mnist/Intro_to_GANs_Exercises.ipynb)


## Project 11:
A simple case study of the batch normalization for normalizing the inputs to layers within the network.

[Batch normalization](https://github.com/sbatururimi/DeepLearning/blob/master/batch-norm/Batch_Normalization_Lesson.ipynb)


## Project 12:
Building a  Deep Convolutional GAN, i.e DCGAN, using convolutional layers in the generator and discriminator. I'll be training DCGAN on the [Street View House Numbers](http://ufldl.stanford.edu/housenumbers/) (SVHN) dataset. These are color images of house numbers collected from Google street view. SVHN images are in color and much more variable than MNIST.



## Project 13:
Training a DCGAN on CelebA to generate new images of human faces

[A generative adversarial network (GAN) trained on the CelebA dataset.](https://github.com/sbatururimi/DeepLearning/blob/master/face_generation/dlnd_face_generation.ipynb)

## Project 14:
Training a semi-supervised GAN for the SVHN (Street View House Numbers) dataset to generate new images and attempt to classify the images with a large proportion of the labels dropped.
[Semi-supervised Learning.](https://github.com/sbatururimi/DeepLearning/blob/master/semi-supervised/semi-supervised_learning_2.ipynb)

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/sbatururimi/DeepLearning/blob/master/LICENSE)