import numpy as np
import open3d as o3d
import os

# import custom classes
from FootEnvironment import FootEnv
from Augmentation import Augmentation
from Preprocessing import Preprocessing
from PcdController import PcdController 
from rl_agent import Agent

# settings
generate_new_clouds = False
down_sample_clouds = False
training = True
continueTraining = False
iters = 3
path = "ply_out/pointcloud_0_00_00.ply"
save_file = "trained_models/PPO/testing"
pc_out_folder = "pc_out"
pc_processed_folder = "ply_out"
pc_org_folder = "original_point_clouds"
size = 1000
result = "Result/PointCloudResult"
voxel_size = 18 ## Default:18 bij size 1000 
distortian = 20 ## Default:20
csv = False ## Default: False
steps = 20 ## Default:20

def Main():

    if generate_new_clouds:
        Augmentation.augment_folder(pc_org_folder,pc_out_folder,distortian,csv)
    
    if down_sample_clouds:
        Preprocessing.convert_folder(pc_out_folder,pc_processed_folder,size,distortian,voxel_size)

    if training:
        Agent.training(pc_processed_folder,save_file,iters,training,continueTraining,steps_max=steps,size_model=size)
    else:
        pcd = o3d.io.read_point_cloud(path)
        Agent.predict(save_file,pcd,pc_processed_folder,result)
        
Main()