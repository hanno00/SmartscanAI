import numpy as np
import open3d as o3d

from Triangulation import Triangulation

def meshing():
    tr = Triangulation()
    tr.convert_pointcloud_to_faces("pointclouds/pc1.xyz")
    print("Done")

meshing()