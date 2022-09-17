# Door-detector-and-Building-classifier
This is a project work done for the Computer Vision and Cognitive System course.
You are invited to read the report for a complete view of the projects and for a more detailed consultation.
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

Initially we tried using ResNet152, with a learning rate of 0.001, 0.005, 0.1, 0.2, and 0.5. We were able to see better performances with the first two values, accordingly we decided to apply those learning rates to smaller ResNet, such as ResNet18, ResNet34 and ResNet50. Finally we tried to optimize the neural networks that gave the best two models by applying the Stochastic Gradient Descent, while in the first tests we made use of Adam method. Table 1 shows the results we achieved at this stage.
As shown here, the best result was achieved with a ResNet34 with 0.005 learning rate value and SGD method.

# Door Detection
First we annotated the dataset (the same one built in precedence) identifying the door in each image, to do that we made use of LabelImg, a graphical annotation tool. The network was pre-trained on COCO and then fine-tuned using google colaboratory on about a thousand images of doors.
As far as the door detection is concerned we used the YOLO network[2], acronym of You Only Look Once, in particular YOLOv2, which is an object detection system aimed for real-time processing. YOLOv2 is less accurate than the region proposal based architectures, but faster because it relies on simpler operations.

# Image retrieval
In order to face the Image Retrieval part we tried two possible approaches, which are different in the type of features extracted from images. The descriptors adopted as first solution are the ORB (Oriented FAST and rotated BRIEF)[3], while for the second solution the descriptors are the features extracted from the neural network used in classification.

# Dataset for retrieval
The dataset used for retrieval is composed by the same images of the previous tasks but this time we applied some image processing algorithm over each image: first of all we resized our images to the fixed size of 400 × 400, then we put them in grey scale, finally we applied an equalized histogram[4] and a bilateral filter[5], whose formula
is given by:

# Image retrival task

In order to extract the features, for this task we modified ResNet. Especially now the net returns features extracted from the last layer before the fully connected one, these features are used as descriptors of the image. In both cases, (ORB descriptor and Neural Network descriptor) as similarity measure, we applied the MSE (Mean Square Error) between the query image and the dataset images.
Before computing a resulting class, there is a query expansion process that consists in some geometric transformations applied to the query image. The geometric transfor-
mations are rotation, perspective transformation and WARP transformation.
After that transformation process the average between the features from the original query image and the features that came from transformed images is computed.
Finally we estimated the MSE between the average features and the features extracted by images of the dataset and the output is the class with the smallest value of MSE.
In Figure 5 is possible to see one of best results of two retrieval algorithm.

# Results

From the conducted experiments, we can conclude that classification via ResNet34 achieves good performance in
testing phase, while the object detection on doors performed by YOLO can still be improved.
For what concern the building recognition, among the three approaches examined, the classification through a neural network proves to be the most satisfactory, since it is also able to give a semantic meaning to the feature it extracts. For the same reason, comparing the other two remaining approaches, the image retrieval carried out by exploiting the feature extracted from the network results more stable then the one carried out via ORB.
Figure 6. Example of result where image retrieval with network
feature does not work.
Figure 7. Example of good result on YOLO detection.
In the next pictures we can see different example of results of our system. In particular, in figure 6 the retrieval with network features does not work, while the results are good for ORB, ResNet and YOLO; figure 7 shows a very good result for door detection, while the retrieval is completely wrong. Also, the result of classification should have been church, but it is understandable why the network could confuse churches and historical buildings; in figure 8 both classification and retrieval are good, while the door detection has a low percentage, probably because that door is different than a classic door; finally in figure 9 the system cannot detect a door and the retrieval with ORB is wrong, but both classification and retrieval with network feature are good.
