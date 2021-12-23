import numpy as np
import pandas as pd
import tensorflow as tf
import open3d as o3d
import os

class Augmentation:
    @staticmethod #class gebonden methode, niet onderdeel van een object
    def load_folder(folder):
        lijst = []
        for file in os.listdir(folder):
            path = os.path.join(folder,file)
            if file.endswith('.ply'):
                pcd = o3d.io.read_point_cloud(path)
                lijst.append(pcd)
        return lijst
    
    @staticmethod
    def augment_folder(folder, save_folder, distortion=20):
        for nummer, file in enumerate(os.listdir(folder)):
            path = os.path.join(folder,file)
            Augmentation.augment_file(path, save_folder,nummer,distortion)  

    @staticmethod
    def augment_file(file, save_folder, number=None, distortion=20):
        if not file.endswith('.csv'):
            print(f"File {file} is not CSV!")
            return
        
        if not os.path.exists(save_folder):
            print(f'Folder {save_folder} cannot be found!')
            return
        
        foot = pd.read_csv(file,dtype=float)
        points = []
        for i in range(len(foot)):
            points.append([foot["X"][i],foot["Y"][i],foot["Z"][i]])
        
        for movement in np.arange(0,0.75,0.15):
            for adding in np.arange(0,1,0.2):
                length = len(points)
                if movement != 0:
                    new_points = tf.random.shuffle(points)
                    length_new = int(movement * len(new_points))
                    new_points_left = new_points[:length_new]
                    new_points_right = new_points[length_new:]
                    new_points_left += tf.random.uniform(new_points_left.shape, -distortion, distortion, dtype=tf.float64)
                    new_points = tf.concat([new_points_right, new_points_left], axis=0)
                    new_points = tf.random.shuffle(new_points)
                else:
                    new_points = tf.random.shuffle(points)
                if adding != 0:
                    length = len(new_points)
                    new_valOrg = int(adding*length)
                    extra_points = tf.random.shuffle(new_points + tf.random.uniform(new_points.shape, -distortion, distortion, dtype=tf.float64))
                    new_points = tf.concat([new_points, extra_points[:new_valOrg]], axis=0)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(new_points)
                path = os.path.join(save_folder,"pointcloud_{}_{:02.0f}_{:02.0f}.ply".format(number,movement*100,adding*100))
                o3d.io.write_point_cloud(path, pcd)
                print(f"Saved pointcloud in {path}.")
    
    @staticmethod
    def upsample(pcd,size,distortion=20):
        if len(np.asarray(pcd.points)) >= size:
            return pcd
        else:
            new_points = tf.random.shuffle(np.asarray(pcd.points))
            new_val_add = size - len(np.asarray(pcd.points))
            extra_points = tf.random.shuffle(new_points + tf.random.uniform(new_points.shape, -distortion, distortion, dtype=tf.float64))
            new_points = tf.concat([new_points, extra_points[:new_val_add]], axis=0)
            pcd.points = o3d.utility.Vector3dVector(new_points)
            return pcd