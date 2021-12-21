import numpy as np
import open3d as o3d
import time

class Triangulation:

    def __init__(self) -> None:
        pass
    
    def convert_pointcloud_to_faces(self, path):
        print("Load a ply point cloud, print it, and render it")
        pcd = self.load_pcd(path)
        self.draw_pcd(pcd, "Original Pointcloud")
    
        print("Downsample the point cloud with a voxel of 0.05")
        pcd = self.voxel_down_sample(pcd,12)
        self.draw_pcd(pcd, "Downsampled")

        print("Compute the normal of the downsampled point cloud")
        pcd = self.computing_normals(pcd,20,5)
        # self.draw_pcd(pcd, "With computed normals")
        # pcd = self.orient_normals(pcd,10)
        # self.compute_cost(pcd,5)

        print("Orient normals")
        for i in range(10):
            pcd = self.orient_normals(pcd,(i + 1) * 5)
            start = time.time()
            print("angle: ", self.compute_cost(pcd,5))
            end = time.time()
            print("time: ", end - start)
            # self.drawPcd(pcd, "Oriented Normals")

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

    def get_points(self, pcd, x):
        print("Get first X points")
        return np.asarray(pcd.points)[:x]
    
    def get_normals(self, pcd, x):
        print("Get first X normals")
        return np.asarray(pcd.normals)[:x]
    
    def compute_cost(self,pcd,knn):
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        amount_of_points = len(np.asarray(pcd.points))
        total_angle = 0
        for i in range(amount_of_points):
            total_angle += self.get_total_angle_from_index(pcd,pcd_tree,knn,i)

        return total_angle / amount_of_points / knn

    def get_total_angle_from_index(self,pcd,pcd_tree,knn,i):
        
        normals = self.get_normals_from_neighbours(pcd,pcd_tree,knn,i)
        anchor_vector = normals[0].tolist()
        
        unit_anchor_vector = anchor_vector / np.linalg.norm(anchor_vector)
        angles = 0
        for j in range(1,knn):
            n_vector = normals[j].tolist()
            unit_n_vector = n_vector / np.linalg.norm(n_vector)
            dot_product = np.dot(unit_anchor_vector, unit_n_vector)
            angle = np.arccos(dot_product)

            # Check for isnan when two vectors are parallel
            if (not np.isnan(angle)):
                angles += angle

        return angles
    
    def get_normals_from_neighbours(self,pcd,pcd_tree,knn,i):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], knn)
        return np.asarray(pcd.normals)[idx]