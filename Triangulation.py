import numpy as np
import open3d as o3d
import time

class Triangulation:

    def __init__(self) -> None:
        pass
    
    def convert_pointcloud_to_faces(self, path):
        print("Load a ply point cloud, print it, and render it")
        pcd = self.load_pcd(path)
        # self.draw_pcd(pcd, "Original Pointcloud")
    
        print("Downsample the point cloud with a voxel of 0.05")
        pcd = self.voxel_down_sample(pcd,12)
        # self.draw_pcd(pcd, "Downsampled")

        print("Recompute the normal of the downsampled point cloud")
        pcd = self.computing_normals(pcd,20,5)
        # self.draw_pcd(pcd, "With computed normals")

        self.print_points(pcd,5)    
        self.print_normals(pcd,5)

        print("Orient normals")
        for i in range(5):
            pcd = self.orient_normals(pcd,(i + 1) * 5)
            self.drawPcd(pcd, "Oriented Normals")

    def voxel_down_sample(self, pcd, size):
        return pcd.voxel_down_sample(voxel_size=size)

    def computing_normals(self, pcd, r, nn):
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=r, max_nn=nn))
        return pcd

    def alpha_meshing(self, pcd, alpha):
        print(f"Doing alpha meshing with alpha={alpha:.3f}")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        mesh.compute_vertex_normals()
        return mesh

    def ball_meshing(self, pcd):
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))
        mesh = mesh.simplify_quadric_decimation(100000)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        return mesh
    
    def poisson_meshing(self, pcd, depth, width, scale):
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=2, linear_fit=False)[0]
        return mesh.crop(pcd.get_axis_aligned_bounding_box())

    def orient_normals(self,pcd,value):
        pcd.orient_normals_consistent_tangent_plane(value)
        return pcd

    def load_pcd(self, path):
        return o3d.io.read_point_cloud(path)

    def draw_pcd(self, pcd, title):
        o3d.visualization.draw_geometries([pcd],title,mesh_show_back_face=True)

    def print_points(self, pcd, x):
        print("Print first X points")
        print(np.asarray(pcd.points)[:x])
    
    def print_normals(self, pcd, x):
        print("Print first X normals")
        print(np.asarray(pcd.normals)[:x])
    