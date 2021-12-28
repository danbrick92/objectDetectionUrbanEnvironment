# Object Detection in an Urban Environment

## Project overview
The goal of this project is to create a Convolutional Neural Network (ConvNet) that can detect objects in an urban environment. Specifically, we are looking to detect pedestrians, cyclists, and other vehicles. This involves being able to detect the bounding boxes of each object in a frame, and also to classify those identified objects.

Object Detection is critical to self-driving cars because it is needed for both safety and navigational needs. It's important to be able to identify both what and where something is on a particular image. For all objects, it's essential to ensure the car does not run into anything and maintains a safe distance. Without object detection (which uses camera images), the car would essentially be blind to the world.

## Set up
[This section should contain a brief description of the steps to follow to run the code for this repository.]

## Dataset
### Dataset analysis
#### Basics
The data in this set come in the form of 640x640 images. Each image looks like the following:

![Unaltered Image](img/base.png?raw=true)

Notice the pedestrians and other vehicles in the image. Although not shown here, there are images in the dataset that contain bicycles.

#### Ground Truth Bounding Boxes
The dataset also includes some other key pieces of information such as the ground truth bounding boxes and the ground truth classes (they correspond with each other).

When the ground truth boxes, classes, and images are overlayed, they create an image that looks like this:

![Overlay Image](img/overlay.png?raw=true)

Notice how each vehicle and pedestrian are outlined with the appropriate bounding boxes, which are color coded by type. 

#### Class Analysis
Based on a random sampling of 10,000 images across the dataset, the following class distribution has been found:

![Class Distribution](img/class_dist.png?raw=true)

On average, most images will contain about 22 bounding boxes in total. 
~78.5% of the bounding boxes in images contain cars. 
~20.9% of the bounding boxes in images contain pedestrians.
~.6 of the remaining boxes are for bicycles.

This is important to recognize as ensuring that bicycles and pedestrians make it into the training process is important. While we could achieve very good results with this dataset, the network could learn to ignore bicylists totally. It could become biased, and this is extremely dangerous.

#### Image Analysis
Although there are many pictures taken on bright blue sunny days, there are many images that are not in those conditions. A list of the most common conditions is shown below: 

![Image Distribution](img/image_dist.png?raw=true)

Some of the images are taken at night. Some are taken in blinding sunlight. Some images contain cars, bicyclists, and pedestrians that are partially occluded. Some images are blurry due to rain or fog. Almost all of the images contain cars and people seen from different angles. 

These are important attributes to consider in order to augment the dataset for training. This should lead to better results and less bias overall. 

### Cross validation
For cross validation, I chose to split the dataset up into three different sections: train, validaton, and test. The rule of thumb for many datasets is to use an 80/20 approach. 80% of the data gets used in training, and the other 20% is used for validation and testing. For the remaining 20%, I opted to split that in half, putting 1/2 in test and 1/2 in validation. 

Although another rule of thumb is to split the dataset up 90/10 if you have a lot of data (and this dataset does), I did not opt to do this. The reason is because I wanted to have a larger representation in the test/validation dataset to prove the results. I would rather feel more confident about the model's performance and generalization ability than it's ability to learn on another 10% of the data. 

In order to accomplish the above, I implemented a function in create_splits.py that splits up the dataset in this way. 

## Training
### Reference experiment
This section should detail the results of the reference experiment. It should includes training metrics and a detailed explanation of the algorithm's performances.

### Improve on the reference
This section should highlight the different strategies you adopted to improve your model. It should contain relevant figures and details of your findings.
