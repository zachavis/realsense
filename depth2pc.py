import numpy as np
import zarr
import sys
from pathlib import *
import cv2
import open3d as o3d
import os
from matplotlib import pyplot as plt
from open3d import geometry
import logging
# [ 1280x720  p[636.435 356.862]  f[636.773 636.773]  Brown Conrady [0 0 0 0 0] ]
# [ 1280x720  p[650.359 359.765]  f[926.46 926.289]  Inverse Brown Conrady [0 0 0 0 0] ]


DEPTH_SCALE = 0.001

class intrinsics:
    # init method or constructor
    def __init__(self, model, height, width, fx, fy, ppx, ppy, coeffs = np.zeros(5)):
        self.height = height
        self.width = width
        self.fx = fx
        self.fy = fy
        self.model = model
        self.ppx = ppx
        self.ppy = ppy
        self.coeffs = coeffs
        model = model
        self.K = np.array([fx,0,ppx,0,fy,ppy,0,0,1]).reshape((3,3))

def convert_depth_frame_to_pointcloud(depth_image, camera_intrinsics, depth_scale = DEPTH_SCALE):
    """
    Convert the depthmap to a 3D point cloud
    Parameters:
    -----------
    depth_frame 	 	 : rs.frame()
                           The depth_frame containing the depth map
    camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
    Return:
    ----------
    x : array
        The x values of the pointcloud in meters
    y : array
        The y values of the pointcloud in meters
    z : array
        The z values of the pointcloud in meters
    """
	
    [height, width] = depth_image.shape

    nx = np.linspace(0, width-1, width)
    ny = np.linspace(0, height-1, height)
    u, v = np.meshgrid(nx, ny)
    x = (u.flatten() - camera_intrinsics.ppx)/camera_intrinsics.fx
    y = (v.flatten() - camera_intrinsics.ppy)/camera_intrinsics.fy

    z = depth_image.flatten() * depth_scale
    x = np.multiply(x,z)
    y = np.multiply(y,z)

    x = x[np.nonzero(z)]
    y = y[np.nonzero(z)]
    z = z[np.nonzero(z)]

    #np.zeros((np.size(x)))
    xyz = np.stack((x,y,z),axis=1)
    #breakpoint()
    return xyz


def load_depth_image(filename : str):
    dimg = zarr.load(filename).astype(np.float32)
    logging.debug('type {}: shape {}'.format(type(dimg), dimg.shape))
    return dimg

def depth2pc(depth_image, depth_intrinsic, trunc_depth = 5):
    
    xyz = convert_depth_frame_to_pointcloud(depth_image, depth_intrinsic)

    # Clip too far:
    xyz = xyz[xyz[:,2]<trunc_depth]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def load_point_clouds(intrinsic, directory, indices = [1], voxel_size=0.0):
    pcds = []
    for idx in indices:
        #breakpoint()
        filename = str(directory / 'depth/frame{:07d}.zarr'.format(idx))
        dimg = load_depth_image(filename)
        pcd = depth2pc(dimg, intrinsic) 
        # pcd = o3d.io.read_point_cloud("../../test_data/ICP/cloud_bin_%d.pcd" %
        #                               i)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd_down.paint_uniform_color(plt.cm.prism(idx/1000)[:-1])
        pcds.append(pcd_down)
    return pcds



def pairwise_registration(source, target):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id])
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph



if __name__ == '__main__':
    depth_intrinsic = intrinsics('Brown Conrady', 1280,720, 636.773, 636.773, 636.435, 356.862 )
    color_intrinsic = intrinsics('Inverse Brown Conrady', 1280,720, 926.46, 926.289, 650.359, 359.765 )
    #breakpoint()



    FILES_DIR = Path(sys.argv[1])
    if len(sys.argv) <= 2:
        FRAME_ID = (np.arange(940)[::15]+1).tolist() #[352, 367, 382]
        breakpoint()
    else:
        FRAME_ID = int(sys.argv[2])
        img = cv2.imread( str(FILES_DIR / 'color/frame{:07d}.jpg'.format(FRAME_ID)) )
        print('type {}: shape {}'.format(type(img), img.shape))
        dimg = zarr.load( str(FILES_DIR / 'depth/frame{:07d}.zarr'.format(FRAME_ID))) #, mode='r')
        print('type {}: shape {}'.format(type(dimg), dimg.shape))


        color_image = img
        depth_image = dimg.astype(np.float32)
        # depth_image = cv2.medianBlur(depth_image, 3)
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs( depth_image, alpha=0.03), cv2.COLORMAP_JET)


        # xyz = convert_depth_frame_to_pointcloud(depth_image, depth_intrinsic)

        # # Clip too far:
        # xyz = xyz[xyz[:,2]<5]

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(xyz)
        # pcd = pcd.voxel_down_sample(voxel_size=0.05)


    voxel_size = 0.05
    pcds_down = load_point_clouds(depth_intrinsic, FILES_DIR, FRAME_ID, voxel_size)

    print("Full registration ...")
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(pcds_down,
                                    max_correspondence_distance_coarse,
                                    max_correspondence_distance_fine)


    print("Optimizing PoseGraph ...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0)
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)

    print("Transform points and display")
    for point_id in range(len(pcds_down)):
        print(pose_graph.nodes[point_id].pose)
        pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
    o3d.visualization.draw_geometries(pcds_down,
                                    zoom=0.3412,
                                    front=[0.4257, -0.2125, -0.8795],
                                    lookat=[2.6172, 2.0475, 1.532],
                                    up=[-0.0694, -0.9768, 0.2024])

    if False:
        if not isinstance(FRAME_ID,list):
            plt.ion()
            plt.show()
            f, axarr = plt.subplots(2,1)
            axarr[0].imshow(color_image)
            axarr[1].imshow(depth_colormap)
            plt.draw()
            plt.pause(0.001)

        o3d.visualization.draw_geometries(pcds_down,
                                    zoom=0.3412,
                                    front=[0.4257, -0.2125, -0.8795],
                                    lookat=[2.6172, 2.0475, 1.532],
                                    #up=[-0.0694, -0.9768, 0.2024])
                                    up=[0, -1, 0])
    #o3d.visualization.draw_geometries([pcd])
    print('moving on')
