from PcdController import PcdController as cloud
import tensorflow as tf
from Augmentation import Augmentation
import open3d as o3d
import numpy as np
import pandas as pd
import os 

class Preprocessing():
    @staticmethod
    def convert_folder(input_folder,output_folder,file_output_size,max_distortion=20,voxel_size=10):
        for file in os.listdir(input_folder):
            path_in = os.path.join(input_folder,file)
            path_out = os.path.join(output_folder,file)
            if file.endswith('.csv'):
                Preprocessing.convert_file(path_in,path_out,file_output_size,max_distortion,voxel_size)

    @staticmethod 
    def convert_file(input_file,output_file,file_output_size,max_distortion=20,voxel_size=10):
        pcd = Preprocessing.csv_to_pcd(input_file)
        length = len(pcd.points)
        if length > file_output_size:
            pcd = Preprocessing.down_sample(pcd,file_output_size)
            print(f'Model of {length} points downsampled to {file_output_size} points')
        elif length < file_output_size:
            pcd = Preprocessing.up_sample(pcd,file_output_size)
            print(f'Model of {length} points upsampled to {file_output_size} points')
        o3d.io.write_point_cloud(output_file.replace('.csv','.ply'), pcd)
        


    @staticmethod
    def up_sample(pcd,size,distortion=20):
        if len(np.asarray(pcd.points)) < size:
            new_points = tf.random.shuffle(np.asarray(pcd.points))
            new_val_add = size - len(np.asarray(pcd.points))
            extra_points = tf.random.shuffle(new_points + tf.random.uniform(new_points.shape, -distortion, distortion, dtype=tf.float64))
            new_points = tf.concat([new_points, extra_points[:new_val_add]], axis=0)
            pcd.points = o3d.utility.Vector3dVector(new_points)
        return pcd

    @staticmethod
    def down_sample(pcd,size,voxel_size=10,distortion=20):
        pcd = pcd.voxel_down_sample(voxel_size)
        if len(pcd.points) > size:
            print('Downsampling failed, retrying with bigger voxel size!')
            pcd = Preprocessing.down_sample(pcd,size,voxel_size*1.5,distortion)
        pcd = Preprocessing.up_sample(pcd,size,distortion)
        return pcd

    @staticmethod
    def csv_to_pcd(csv):
        if not csv.endswith('.csv'):
            print(f'File {csv} is not a CSV file!')
            return None
        foot = pd.read_csv(csv,dtype=float)
        points = []
        for i in range(len(foot)):
            points.append([foot["X"][i],foot["Y"][i],foot["Z"][i]])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd