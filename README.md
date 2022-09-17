# Door-detector-and-Building-classifier
This is a project work done for the Computer Vision and Cognitive System course.
# CV Project Work - UNIMORE 2020/2021

### The Team:"

- Benedetta Fabrizi ([LinkedIn](https://www.linkedin.com/in/benedetta-fabrizi-54b7971b0) - [GitHub](https://github.com/BerniRubble))
- Emanuele Bianchi ([LinkedIn](https://www.linkedin.com/in/emanuele-bianchi240497/) - [GitHub](https://github.com/Manu2497))

## The project 🔍

Purpose of this work is the development of a frame-work for buildings recognition and the detection of their entrance. To this end, two types of neural networks are used:
one for the classification of buildings and one for door detection. As regards the classification task, a comparisonwith other approaches based on image retrieval was also made, always with the goal of being able to classify the building, making use of image processing and, therefore, applying filters and image transformations

# 1. Introduction
The use of neural networks can improve everyday life
quality of everyone, but mostly important they can be used
for helping people with particular deseas.

# 2.Buildings Classification
In this Section we describe the dataseta and the chosen Artificial Neural Network for the classification of buildings.
Finally, a description of the tests run for choosing the best model are reported.

# 2.1. Dataset for ResNet
Since the impossibility to find a dataset about buildings that fit the project requirements, we built our personal dataset by picking the images from different stock websites.Then, we have arranged them in five different folders, corresponding to the classes defined in the project: house, flat complex, church, historical buildings/monument, shops. After having filled the dataset, we cleaned it from those images not containing useful information. The resulting dataset is balanced, with each class approximately having the same number of photos. Since we built or own dataset, it is not very large. Indeed, the total amount of images is 1300.
During classification, both in training phase and test phase, the artificial neural network will read directly the folders name and take them as classes name.
# 2.2. Choice of model
For classification purpose, we made use of Residual Network, or ResNet[1], which is very popular for image classification. In order to make it compatible with our final goal,we modified the last layer before the fully connected one,also providing it with an additional output: therefore our network returns a matrix [1, 5], where 5 is the number of classes, and the feature vector that we will use in section 4.
Furthermore, we did a fine-tuning freezing the first layers and retraining just the last 19 layers.
Since ResNet architecture may vary in quantity of layers, we made some tests to find the best model to fit our data.
To do so, we used the early stopping technique, plotting both train and validation loss and saving the model with the
lower loss value. At first we set up a patience of 5 epochs, afterwards we increased it to 20 if the analysis of the curve
showed that the model was not overfitting and had a decreasing trend.
Figure 1 shows an example of early stopping method
with 5 epoches patience in which the validation loss has a downward trend, whereby the patience could be raised to 20.
Figure 2 
