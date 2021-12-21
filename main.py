import numpy as np
import open3d as o3d

from PcdController import PcdController as pc

def testing():
    pcd = pc.load_pcd("pointclouds/pc1.xyz")
    pc.draw_pcd(pcd,"Original")

    pcd = pc.voxel_down_sample(pcd,12)
    pc.draw_pcd(pcd,"Downsampled")

    pcd = pc.computing_normals(pcd,20,5)
    pc.draw_pcd(pcd,"Normals")

    pcd = pc.orient_normals(pcd, 10)
    pc.draw_pcd(pcd,"Oriented normals")
    
    print("Computed cost: ", pc.compute_cost(pcd,5))
        
testing()