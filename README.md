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
### Experiment 1: Reference Model
#### Overview
The initial reference model did not have great performance.

#### mAP
For the Mean Average Precision (mAP) @ 50 IOU, it achieved a precision of .08. Considering an optimal mAP is 1.0, this is not a good start.
![mAP Run 0](img/run_0_map.png?raw=true)

#### Learning Rate
For the learning rate, the model started with a low learning rate until it hit about 2000 steps. At this point, the learning rate grew to about .04 and then started to decrease steadily until it hit 0 at 25000 steps.

![Learning Rate Run 0](img/run_0_learningrate.png?raw=true)

#### Loss
In terms of loss, there were three recorded metrics: the classification loss (ability to correctly identify object class), the localization loss (for bounding boxes), and the regularization loss (loss due to the regularizer). I only show the classification and localization loss here.

The classification loss was at about .5 by the end of 24000 steps in the training, but performed worse at .59 on the validation data. This shows that there was some amount of overfitting on the data since we did not generalize well to the test set. This could be due to many things like poor regularization, too high a learning rate, bad architectural decisions, or lack of augmentation. 

![Classification Loss Run 0](img/run_0_loss_classification.png?raw=true)

A very similar story can be told for the localization loss. It was .4 on the training set by 24000 steps, and .52 on the validation set. Again, this could be due to any of the factors I mentioned above.

![Localization Loss Run 0](img/run_0_loss_localization.png?raw=true)


### Experiment 2: Model with Augmentations, Tuned Parameters
#### Overview
What's changed?
-Added the following augmentations:
	-random_black_patches
	-random_adjust_brightness
- Changed depth of box predictor from 256 to 512

Based on the poor performance of the last network, I observed that there might be some new agumentations needed. I also noticed that there was a fairly steady decline in loss in the last network. I decided to experiment with adding more layers. 

Overall, the new network had subpar performance. In retrospect, I should have added more steps since it did not appear that the last version of the model converged.

#### mAP
For the mAP score, even with the large box with an IOU threshold of .6, I only received a score of .017. This is actually a huge downgrade from the performance of the last network. 
![mAP Run 1](img/run_1_map.png?raw=true)

#### Learning Rate
The learning rate is the same as it was in the Reference Model.
![Learning Rate Run 1](img/run_1_learningrate.png?raw=true)

#### Loss
The classification loss over time did decrease, but only slighly. After the last 50% of the training time, there was basically no improvement.

On the other hand, the localization loss experienced a good steady decrease over time, as did the regularization loss.

Interestingly enough, the overloss for the validation data and training data were roughly the same. This does indicate to me that the model may have benefited from a longer training period since it experienced no signs of overfitting.
![Loss Run 1](img/run_1_loss.png?raw=true)

### Experiment 3: Changing to EfficientNet D1 Network, Longer Training Time
#### Overview
What's changed?
- Changed to a new network "ssd_efficientnet-b1_bifpn_keras"
- Set l2 regularizers from .0004 to .0002
- Changed from ReLU activation to SWISH
- Changed from 25000 steps to 50000 steps

After doing some research, I found that YOLOv4, YOLOR, and EfficientNets are the current SOTA for object detection. Based on availability in the model garden, I opted to select EfficientNet as the new architecture. I also took notice of the lack of convergence on the previous models. Both models had been improving when the training was cut short at 25000 steps. I decided to double the amount of time in order to help with this problem. Finally, I also switched to using SWISH instead of ReLU for activation due to SWISH outperforming ReLU on many tasks.

Overall, the new network had much improved performance.

#### mAP
For the mAP score, even with the large box with an IOU threshold of .6, I a score of .3976, which is much more of an improvement. 
![mAP Run 2](img/run_2_map.png?raw=true)

After doing a sample visualization, I noticed that the network is not as good with detecting pedestrians that are far off in the distance. I believe some improvements will need to be made to accomodate this and receive a higher mAP score.

#### Learning Rate
The learning rate, due to the longer runtimes, had more of a slow decline over the 50000 steps.
![Learning Rate Run 2](img/run_2_learningrate.png?raw=true)

#### Loss
The classification loss over time did decrease quite steady, starting around 1.2 and ending at .22. 
The localization loss was also very similar, experiencing a near identical decrease. 

Overall, there is definitely some overfitting going on here for both classification and localization. Both the classification loss and localization loss were about 1.5 times larger on the validation data. 
![Loss Run 2](img/run_2_loss.png?raw=true)

### Experiment 4: Longer Training Times, Increased Regularization and Learning Rate, Added More Augmentations
#### Overview
What's changed?
- Set the number of steps from 50000 to 60000
- Set l2 regularizers from .0002 to .0003
- Added augmentation random_patch_gaussian
- Added augmentation random_image_scale

In the last network, I saw significantly increased performance but some signs of overfitting. I also saw that there may have been more potential for more learning. To balance these two needs, I made some changes in regularization and the number of steps. I recognized that my network was performing poorly on objects farther away, and did not account for blurred images. To account for these, I added random_patch_gaussian for blurriness and random_image_scale to accomodate smaller objects. 

Overall, the new network had mildly improved performance, but showed significant signs of overfitting. 

#### mAP
The mAP score with the large box and an IOU threshold of .6 was .4237. This is a modest improvement over the last model.
![mAP Run 3](img/run_3_map.png?raw=true)

Upon observation of some sample animations, it does still appear to be having a lot of trouble with distant pedestrians.

#### Learning Rate
The learning rate was the same as the last run. 
![Learning Rate Run 3](img/run_3_learningrate.png?raw=true)

#### Loss
While the training loss for both classification and localization loss were lower than last time, it performed worse on the validation data. (Both classification and localization loss for validation data are about twice as bad as the losses for training).
![Loss Run 3](img/run_3_loss.png?raw=true)
