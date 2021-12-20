import numpy as np
import pandas as pd
import tensorflow as tf
import random as rd
import open3d as o3d
import os

par = 20
folder = "OriginalPointClouds"
for number,file in enumerate(os.listdir(folder)):
    if file.endswith(".csv"):
        foot = pd.read_csv(os.path.join(folder,file))

        points = []
        for i in range(len(foot)):
            points.append([foot["X"][i],foot["Y"][i],foot["Z"][i]])

        for movement in np.arange(0,0.75,0.15):
            for adding in np.arange(0,1,0.2):
                length = len(points)
                if movement != 0:
                    NewPoints = tf.random.shuffle(points)
                    lengthNew = int(movement * len(NewPoints))
                    NewPointsLeft = NewPoints[:lengthNew]
                    NewPointsRight = NewPoints[lengthNew:]
                    NewPointsLeft += tf.random.uniform(NewPointsLeft.shape, -par, par, dtype=tf.float64)
                    NewPoints = tf.concat([NewPointsRight, NewPointsLeft], axis=0)
                    NewPoints = tf.random.shuffle(NewPoints)
                else:
                    NewPoints = tf.random.shuffle(points)
                if adding != 0:
                    length = len(NewPoints)
                    new_valOrg = int(adding*length)
                    ExtraPoints = tf.random.shuffle(NewPoints + tf.random.uniform(NewPoints.shape, -par, par, dtype=tf.float64))
                    NewPoints = tf.concat([NewPoints, ExtraPoints[:new_valOrg]], axis=0)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(NewPoints)
                o3d.io.write_point_cloud("PointClouds/pointcloud{}_{:.2f}_{:.2f}.ply".format(number,movement,adding), pcd)