# This is a sample Python script.
import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans, DBSCAN, OPTICS
import matplotlib.pyplot as plt
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

    # pcd = o3d.io.read_point_cloud("pvso/pointcloud/internet_pcd.ply",format='ply')
    pcd = o3d.io.read_point_cloud("pvso/pointcloud/output_big4.ply", format='ply')

    # Convert PCD point cloud to PLY format in memory
    # ply_str = io.StringIO()
    # o3d.io.write_point_cloud(ply_str, pcd, format='ply', write_ascii=True)
    # ply_str.seek(0)
    # ply_data = ply_str.read()
    # o3d.io.write_point_cloud("pvso/pointcloud/outputbig4_bin.ply", pcd, write_ascii=True)
    # pcd = o3d.io.read_point_cloud("pvso/pointcloud/outputbig4_bin.ply")

    print(pcd)
    # print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd])

    cl, inliers = pcd.remove_statistical_outlier(nb_neighbors=20,
                                                 std_ratio=2.0)
    # plane_model, inliers = pcd.segment_plane(distance_threshold=0.03, ransac_n=3, num_iterations=1000)
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    # inlier_cloud.paint_uniform_color([1, 0, 0])
    outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
    o3d.visualization.draw_geometries([inlier_cloud])

    points = np.asarray(inlier_cloud.points).copy()
    scaled_points = StandardScaler().fit_transform(points)
    # Clustering:
    model = KMeans(n_clusters=10)
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
    inlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([inlier_cloud])

    # o3d.io.write_point_cloud("pvso/pointcloud/output_big1_inliers.ply", inlier_cloud, write_ascii=True)
    # pcd = o3d.io.read_point_cloud("pvso/pointcloud/output_big1_inliers.ply")
    # pcd = inlier_cloud
    # points = np.asarray(pcd.points)
    # # Define k-means parameters
    # max_clusters = 10
    # max_iters = 100
    #
    # # Perform k-means with a range of k values
    # inertias = []
    # for k in range(1, max_clusters + 1):
    #     kmeans = KMeans(n_clusters=k, max_iter=max_iters).fit(points)
    #     inertias.append(kmeans.inertia_)
    #
    # # Use elbow method to estimate optimal number of clusters
    # diff = np.diff(inertias)
    # plt.plot(range(1, max_clusters + 1), inertias, '-o')
    # plt.plot(range(1, max_clusters), diff, '-o')
    # plt.xlabel('Number of clusters (k)')
    # plt.ylabel('Inertia')
    # plt.title('Inertia vs. number of clusters')
    # plt.xticks(range(1, max_clusters + 1))
    # plt.show()
    #
    # # Choose optimal number of clusters and perform k-means again
    # optimal_k = np.argmin(diff) + 1
    # kmeans = KMeans(n_clusters=optimal_k, max_iter=max_iters).fit(points)
    # print(optimal_k)
    # labels = kmeans.labels_
    # if optimal_k > 1:
    #     colors = plt.get_cmap("tab20")(labels / (optimal_k - 1))
    # else:
    #     colors = plt.get_cmap("tab20")(labels)
    # colors = colors[:, :3]
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    # # Visualize point cloud with colors assigned by cluster
    # o3d.visualization.draw_geometries([pcd])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
