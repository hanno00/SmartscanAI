import numpy as np
import open3d as o3d

class PcdController:
    @staticmethod
    def voxel_down_sample(pcd, size):
        return pcd.voxel_down_sample(voxel_size=size)

    @staticmethod
    def computing_normals(pcd, r, nn):
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=r, max_nn=nn))
        return pcd

    @staticmethod
    def alpha_meshing(pcd, alpha):
        print(f"Doing alpha meshing with alpha={alpha:.3f}")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        mesh.compute_vertex_normals()
        return mesh

    @staticmethod
    def ball_meshing(pcd):
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
    
    @staticmethod
    def poisson_meshing(pcd, depth, width, scale):
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=2, linear_fit=False)[0]
        return mesh.crop(pcd.get_axis_aligned_bounding_box())

    @staticmethod
    def orient_normals(pcd,value):
        pcd.orient_normals_consistent_tangent_plane(value)
        return pcd

    @staticmethod
    def load_pcd(path):
        return o3d.io.read_point_cloud(path)

    @staticmethod
    def draw_pcd(pcd, title):
        o3d.visualization.draw_geometries([pcd],title,mesh_show_back_face=True)

    @staticmethod
    def get_points(pcd, x):
        print(f"Get first {x} points")
        return np.asarray(pcd.points)[:x]
    
    @staticmethod
    def get_normals(pcd, x):
        print(f"Get first {x} normals")
        return np.asarray(pcd.normals)[:x]
    
    @staticmethod
    def compute_cost(pcd,knn):
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        amount_of_points = len(np.asarray(pcd.points))
        total_angle = 0
        for i in range(amount_of_points):
            total_angle += PcdController.__get_total_angle_from_index(pcd,pcd_tree,knn,i)

        return total_angle / amount_of_points / knn

    @staticmethod
    def __get_total_angle_from_index(pcd,pcd_tree,knn,i):
        
        normals = PcdController.__get_normals_from_neighbours(pcd,pcd_tree,knn,i)
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

    @staticmethod
    def __get_normals_from_neighbours(pcd,pcd_tree,knn,i):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], knn)
        return np.asarray(pcd.normals)[idx]