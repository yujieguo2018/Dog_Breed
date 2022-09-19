# Dog_Breed

## Project Overview 

This project uses Convolutional Neural Networks (CNNs)! Given an image of a dog, the algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed.

##  Technique/Methodology

The core of this project is to understand how to apply CNN in image classification by experimenting with different model structure(e.g Convolutional layer, MaxPooling layer, Dense layer), training techniques, and testing their performance. We both try to put together a model from scratch or use Transfer Learning. The main metrics we evaluate our model is the accuracy -- how well the model could identify the dog breed.

## Data Processing

This is a deep learning project, we don't have much data processing comparing to a non-deep learning machine learning project because the algorithm would extract features from the image instead of we maually do feature engineering for the model. But there are some data processing related to picture displaying and the data format that works with Tensorflow.

## Model Evaluation and Validation

We trained the models and selected the best model with the best validation loss, use the weights in this model to make prediction in the test dataset.

## Summary

1. We used ResNet50 model with weights = 'imagenet', achieve 99% accuray.( a pre-trained [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) model to detect dogs in images.  Our first line of code downloads the ResNet-50 model, along with weights that have been trained on [ImageNet](http://www.image-net.org/), a very large, very popular dataset used for image classification and other vision tasks.  ImageNet contains over 10 million URLs, each linking to an image containing an object from one of [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).  Given an image, this pre-trained ResNet-50 model returns a prediction (derived from the available categories in ImageNet) for the object that is contained in the image.)
2. We used the model we created from scratch, only achieved 7.77% accuracy.
3. We used transfer learning to create a CNN using  VGG-16 model as a fixed feature extractor, achieved 39.2% accuray.
4. We used transfer learning to create a CNN using pre-trained ResNet-50 bottleneck features, this highly improved our accuracy to be 80.0%.

There are still some shortcoming in my final model, like it mix up American water spaniel and Boykin spaniel, and if supplied an image of a human, the identified resembling dog breed isn't very accurate, I think this is because of underfitting and I should add extra layers to extract more features and train it on more epoches to improve its performance.




