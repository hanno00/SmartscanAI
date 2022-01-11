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

### main.py

Tekst hierover

### Augmentation.py

Tekst over augmentatie

### Preprocessing.py

Tekst hierover

### [PcdController.py](https://github.com/hanno00/SmartscanAI/blob/main/PcdController.py)

This is a static class which can be used to make all kind of changes to a pointcloud. Most of them directly use the [open3d](http://www.open3d.org/) library to do these changes. The cost that is being used in the reinforcement learning also gets computed here.

### FootEnvironment.py

Tekst hierover

### rl_agent.py

Tekst hierover
