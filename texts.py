import open3d as o3d
import numpy as np

txt_path = r'C:\Users\86159\PycharmProjects\lidar\Pointnet_Pointnet2_pytorch-master\data\modelnet40_normal_resampled\car\car_0001.txt'
# 通过numpy读取txt点云
pcd = np.genfromtxt(txt_path, delimiter=",")

pcd_vector = o3d.geometry.PointCloud()
# 加载点坐标
print(pcd_vector.points)
pcd_vector.points = o3d.utility.Vector3dVector(pcd[:, :3])
o3d.visualization.draw_geometries([pcd_vector])
