import numpy as np
import pandas as pd
import tensorflow as tf
import random as rd
import open3d as o3d

data = pd.read_csv("usertest1_pc.csv")

points = []
for i in range(len(data)):
    points.append([data["X"][i],data["Y"][i],data["Z"][i]])

par = 5

pointsOrg = tf.random.shuffle(points)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pointsOrg)
o3d.io.write_point_cloud("PointClouds/pc1.ply", pcd)
#np.savetxt('PointClouds/pc1.csv', pointsOrg, delimiter=',',header='X,Y,Z',comments='') 

pointsMov = tf.random.shuffle(points)
pointsMov += tf.random.uniform(pointsMov.shape, -par, par, dtype=tf.float64)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pointsMov)
o3d.io.write_point_cloud("PointClouds/pc2.ply", pcd)
#np.savetxt('PointClouds/pc2.csv', pointsMov, delimiter=',',header='X,Y,Z',comments='') 

ranges = range(3,21)
percentOrg = 0.1
percentMov = 0.1
for i in ranges:
    length = len(points)
    if (i % 2) == 1:
        new_valOrg = int(percentOrg*length)
        ExtraOrg = tf.random.shuffle(pointsOrg + tf.random.uniform(pointsOrg.shape, -par, par, dtype=tf.float64))
        NewPCOrg = tf.concat([pointsOrg, ExtraOrg[:new_valOrg]], axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(NewPCOrg)
        o3d.io.write_point_cloud("PointClouds/pc"+ str(i) +".ply", pcd)
        #np.savetxt('PointClouds/pc'+str(i)+'.csv', NewPCOrg, delimiter=',',header='X,Y,Z',comments='') 
        percentOrg += 0.1
    else:
        new_valMov = int(percentMov*length)
        ExtraMov = tf.random.shuffle(pointsMov + tf.random.uniform(pointsMov.shape, -par, par, dtype=tf.float64))
        NewPCMov = tf.concat([pointsMov, ExtraMov[:new_valMov]], axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(NewPCMov)
        o3d.io.write_point_cloud("PointClouds/pc"+ str(i) +".ply", pcd)
        #np.savetxt('PointClouds/pc'+str(i)+'.csv', NewPCMov, delimiter=',',header='X,Y,Z',comments='') 
        percentMov += 0.1