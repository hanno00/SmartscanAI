
# Smartscan AI

## General information

This Github-project can be used to train a reinforcement learning agent, in order to make pointclouds smoother. It does this by moving the points in the pointcloud around, and getting the average difference between all normals.

## Project structure

The project consists of three parts.

### Data Augmentation

Data augmentation is responsible for generating a large dataset from only a few initial pointclouds. It does this by moving around the points in the initial pointclouds or adding more points to them.

### Preprocessing

Preprocessing prepares all the pointclouds created in the previous step for training. The reinforcement learning algorithm that is being used does not allow for dynamic test sizing. Therefore we mold all pointclouds to be of a certain adjustable size.

### Learning

The reinforcement learning is done by the [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/) library. The library has been implemented and the reinforcement learning agent can move all the points in the pointclouds around. The reward function is as follows:

* Positive feedback if the pointcloud is smoother. (This is calculated by taking the average angle difference between the normal of a points and its neighbours, summed up for all points)
* Negative feedback for the distance a point is moved away from its original position. This is to prevent the model from moving all points into a single point or a ball, since those shapes are the smoothest.

## Classes

In this section the purpose and most important functions of the classes are explained.

### [main.py](https://github.com/hanno00/SmartscanAI/blob/main/main.py)

This file runs the entire project and has all the options available for tweaking. All the options are explained below:

| Option                         | Default Value                        | Description                                                                                                                                                                                                                      |
|--------------------------------|--------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Augmentation**                   |                                      |                                                                                                                                                                                                                                  |
| augment_new_pointclouds        |                                 True | When this option is turned on, the program will generate 25 variant   pointclouds for each pointcloud in the original_pointclouds folder                                                                                         |
| original_pointclouds           |                 original_pointclouds | This is the folder path in which the original pointclouds are placed                                                                                                                                                             |
| augmented_pointclouds          |                augmented_pointclouds | This is the folder path in which the augmented pointclouds will be placed                                                                                                                                                        |
| save_augmented_as_csv          |                                False | This option specifies whether the augmented pointclouds will be saves as   .ply files (False), or as .csv files (True)                                                                                                           |
| **Preprocessing**                  |                                      |                                                                                                                                                                                                                                  |
| preprocess_pointclouds_to_size |                                 True | When this option is turned on, the program will process each augmented   pointcloud to make them the size specified in preprocessing_size                                                                                        |
| preprocessing_size             |                                 1000 | This size determines the amount of points there will be in all the   pointclouds. More points mean more details, but longer training times                                                                                       |
| downsample_voxelsize           |                                   18 | The voxelsize is used to downsample the pointclouds to their prefered   size. Learn more   [here](https://adioshun.gitbooks.io/pcl/content/Tutorial/Filtering/pcl-cpp-downsampling-a-pointcloud-using-a-voxelgrid-filter.html)   |
| preprocessed_pointclouds       |             preprocessed_pointclouds | This is the folder path in which the preprocessed pointclouds will be   placed                                                                                                                                                   |
| distortion                     |                                   20 | This is the amount of movement allowed when augmenting en preprocessing.   This value should be somewhere close to the downsample_voxelsize. A lower   value means less noise, but if it is too low, points may start to overlap |
| **Training**                       |                                      |                                                                                                                                                                                                                                  |
| train                          |                                 True | When this option is turned on, the program will start training                                                                                                                                                                   |
| continue_training              |                                False | If turned on, the programm will continue the training with a pretrained   model, if turned off, it will start from scratch                                                                                                       |
| training_iterations            |                                  100 | This is the amount of training iterations                                                                                                                                                                                        |
| steps_per_iteration            |                                   20 | This is the amount of steps per iteration                                                                                                                                                                                        |
| trained_model_filepath         |             trained_models/PPO/model | This is the folder path in which the trained model will be saved. In the   default case in the folder "trained_models/PPO/", the trained model   will be saved as "model"                                                        |
| verbose_training               |                                False | This option specifies whether the program will print out its progress   while training                                                                                                                                           |
| **Predicting**                     |                                      |                                                                                                                                                                                                                                  |
| predict                        |                                 True | When this option is turned on, the program will predict using the   pre-trained model                                                                                                                                            |
| prediction_input_filepath      | preprocessed_pointclouds/example.ply | This is the filepath of the pointcloud which will be used as input for   the prediction                                                                                                                                          |
| predicted_pointclouds          |             result/pointcloud_result | This is the filepath of the resulting pointcloud. In the default case in   the folder "result/", the predicted model will be saved as   "pointcloud_result"                                                                      |
| verbose_prediction             |                                False | This option specifies whether the program will print out its progress   while predicting                                                                                                                                         |

### [Augmentation.py](https://github.com/hanno00/SmartscanAI/blob/main/Augmentation.py)

The Augmentation class is intended to enlarge the dataset. By loading a csv or ply file, the points of the point clouds can be moved. In addition, points can be added. These are generated at an arbitrary distance (variable) from the original points.

### [Preprocessing.py](https://github.com/hanno00/SmartscanAI/blob/main/Preprocessing.py)

The Preprocessing class is intended to get the point clouds to the desired number of points. The desired number of points is variable, but must always remain the same within a training session.

### [PcdController.py](https://github.com/hanno00/SmartscanAI/blob/main/PcdController.py)

This is a static class which can be used to make all kind of changes to a pointcloud. Most of them directly use the [open3d](http://www.open3d.org/) library to do these changes. The cost that is being used in the reinforcement learning also gets computed here.

### [FootEnvironment.py](https://github.com/hanno00/SmartscanAI/blob/main/FootEnvironment.py)

This class houses the environment in which the Reinforcment Learning Agent works. This class is responsible for providing the agent with pointclouds, apply its actions, calculating the effectiveness of the actions and providing that feedback to the agent. This class randomly selects the next pointcloud (.ply files generated by the preprocessing step) from its source folder, but optionally the next pointcloud can be provided as an argument in the reset function. For the environment, its important that the size of the pointclouds matches the size expected by the class (default 3500).

## Installation Guide

To run this program, follow these steps:

1. Download the Git repository to your machine.

2. Use the following command in your Anaconda Prompt to generate a fitting Conda environment.

    ` conda env create -f smartscan_env.yml `

3. Activate your conda environment.

4. Before you run the program, go to main.py and check out at all the settings.

5. Set everything up as you want (options explained above).

6. Don't forget to create all the folders you have specified in the options.

7. Run main.py.
