# This is a sample Python script.
import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans, DBSCAN, OPTICS, MeanShift
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
import io


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    pcd = o3d.io.read_point_cloud("pvso/pointcloud/internet_pcd.ply",format='ply')
    # pcd = o3d.io.read_point_cloud("pvso/pointcloud/output_big6.ply", format='ply')

    # Convert PCD point cloud to PLY format in memory
    # ply_str = io.StringIO()
    # o3d.io.write_point_cloud(ply_str, pcd, format='ply', write_ascii=True)
    # ply_str.seek(0)
    # ply_data = ply_str.read()
    # o3d.io.write_point_cloud("pvso/pointcloud/outputbig4_bin.ply", pcd, write_ascii=True)
    # pcd = o3d.io.read_point_cloud("pvso/pointcloud/outputbig4_bin.ply")

    print(pcd)
    # print(np.asarray(pcd.points))
    # o3d.visualization.draw_geometries([pcd])

    cl, inliers = pcd.remove_statistical_outlier(nb_neighbors=20,
                                                 std_ratio=2.0)
    # plane_model, inliers = pcd.segment_plane(distance_threshold=0.03, ransac_n=3, num_iterations=1000)
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    # inlier_cloud.paint_uniform_color([1, 0, 0])
    outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
    # o3d.visualization.draw_geometries([inlier_cloud])

    pcd = inlier_cloud
    points = np.asarray(pcd.points)

    # Normalisation:
    scaled_points = StandardScaler().fit_transform(points)

    # Clustering:
    model = DBSCAN(eps=0.15, min_samples=10)
    # model = KMeans(n_clusters=4)
    model.fit(scaled_points)

    # Get labels:
    labels = model.labels_
    # Get the number of colors:
    n_clusters = len(set(labels))

    # Mapping the labels classes to a color map:
    colors = plt.get_cmap("tab20")(labels / (n_clusters if n_clusters > 0 else 1))
    # Attribute to noise the black color:
    colors[labels < 0] = 0
    # Update points colors:
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # Display:
    o3d.visualization.draw_geometries([pcd])