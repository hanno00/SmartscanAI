import numpy as np
import open3d as o3d

class Triangulation:
    
    def convertPointcloudToFaces(path):
        print("Load a ply point cloud, print it, and render it")
        pcd = o3d.io.read_point_cloud(path)
        print(pcd)
        print(np.asarray(pcd.points))
        o3d.visualization.draw_geometries([pcd])

        # print("Downsample the point cloud with a voxel of 0.05")
        # downpcd = pcd.voxel_down_sample(voxel_size=12)
        # o3d.visualization.draw_geometries([downpcd])
        downpcd = pcd

        print("Recompute the normal of the downsampled point cloud")
        # downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        #     radius=0.1, max_nn=30))
        downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=20, max_nn=5))
        downpcd.orient_normals_consistent_tangent_plane(100)
        o3d.visualization.draw_geometries([downpcd])

        print("Print a normal vector of the 0th point")
        print(downpcd.normals[0])
        print("Print the normal vectors of the first 10 points")
        print(np.asarray(downpcd.normals)[:10, :])
        print("")

        alpha = 20
        print(f"alpha={alpha:.3f}")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(downpcd, alpha)
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
        output_path="plyOutput/"
        o3d.io.write_triangle_mesh(output_path+"alphashape1.ply", mesh)
        print(mesh)
        print(type(mesh))

        # distances = downpcd.compute_nearest_neighbor_distance()
        # avg_dist = np.mean(distances)
        # radius = 4 * avg_dist

        # bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(downpcd,o3d.utility.DoubleVector([radius, radius * 2]))
        # dec_mesh = bpa_mesh.simplify_quadric_decimation(100000)
        # dec_mesh.remove_degenerate_triangles()
        # dec_mesh.remove_duplicated_triangles()
        # dec_mesh.remove_duplicated_vertices()
        # dec_mesh.remove_non_manifold_edges()

        # o3d.visualization.draw_geometries([bpa_mesh])

        # # for i in range(5):
        #     print("i", i,)
        #     poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(downpcd, depth=8, width=0, scale=2 + (i*0.5), linear_fit=False)[0]
        #     bbox = pcd.get_axis_aligned_bounding_box()
        #     p_mesh_crop = poisson_mesh.crop(bbox)
        #     output_path="plyOutput/"
        #     o3d.io.write_triangle_mesh(output_path+"p_mesh_c.ply", p_mesh_crop)
        #     o3d.visualization.draw_geometries([p_mesh_crop])