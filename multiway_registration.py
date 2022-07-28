import numpy as np
import zarr
import sys
from pathlib import *
import cv2
import open3d as o3d
import os
import platform
from matplotlib import pyplot as plt
from open3d import geometry
from open3d import io as o3dio
import logging

USE_RS2 = False
if platform.system() == 'Windows' or platform.system() == 'Linux':
    USE_RS2 = True
    import pyrealsense2 as rs2
    # post-processing goes here
    #align = rs2.align(align_to)

    decimate = rs2.decimation_filter(2.0)
    threshold = rs2.threshold_filter(0.1, 4.0)
    # depth2disparity = rs2.disparity_transform()
    # spatial = rs2.spatial_filter(0.5, 20.0, 2.0, 0.0)
    # temporal = rs2.temporal_filter(0.0, 100.0, 3)
    # disparity2depth = rs2.disparity_transform(False)
else:
    decimate = lambda x : x
    threshold = lambda x : x


    # align_to = rs2.stream.color
    # decimated = decimate.process(frames).as_frameset()
    # thresholded = threshold.process(decimated).as_frameset()
    # disparity = depth2disparity.process(thresholded).as_frameset()
    # spatial = spatial.process(disparity).as_frameset()
    # # temporal = self.temporal.process(spatial).as_frameset() # TODO: re-enable
    # postprocessed = disparity2depth.process(spatial).as_frameset()


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

    z = depth_image.flatten() #* depth_scale
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
    return dimg / 1000 #* DEPTH_SCALE

def depth2pc(depth_image, depth_intrinsic, trunc_depth = 3):
    
    xyz = convert_depth_frame_to_pointcloud(depth_image, depth_intrinsic)

    # Clip too far:
    #xyz = xyz[xyz[:,2]] <trunc_depth]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def load_point_clouds(intrinsic, directory, indices = [1], voxel_size=0.0):
    pcds = []
    for idx in indices:
        #breakpoint()
        filename = str(directory / 'depth/frame{:07d}.zarr'.format(idx))
        dimg = load_depth_image(filename)
        # dimg = decimate.process(dimg)
        # breakpoint()
        # dimg = threshold.process(dimg)
        dimg[dimg < .1] = 0
        dimg[dimg > 4] = 0
        # breakpoint()
        pcd = depth2pc(dimg, intrinsic) 
        # pcd = o3d.io.read_point_cloud("../../test_data/ICP/cloud_bin_%d.pcd" %
        #                               i)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd_down.paint_uniform_color(plt.cm.prism(idx/1000)[:-1])
        pcds.append(pcd_down)
    return pcds

def load_RGBDs(intrinsic, directory, indices = [1]):
    rgbds = []
    for idx in indices:
        #breakpoint()
        filename = str(directory / 'depth/frame{:07d}.zarr'.format(idx))
        dimg = load_depth_image(filename) * DEPTH_SCALE
        #pcd = depth2pc(dimg, intrinsic)
        filename = str(directory / 'color/frame{:07d}.jpg'.format(idx))
        rgb = o3d.io.read_image( filename )
        # pcd = o3d.io.read_point_cloud("../../test_data/ICP/cloud_bin_%d.pcd" %
        #                               i)
        dimg = o3d.geometry.Image(dimg)
        #breakpoint()
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, dimg) # default scale is 1000, trunc = 3
        rgbds.append(rgbd)
    return rgbds



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
    #breakpoint()
    return transformation_icp, information_icp


def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)

    # Register all for loop closure:
    for source_id in range(n_pcds):
        for target_id in range(n_pcds):
            if source_id == target_id:
                continue
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                        target_id,
                                                        np.eye(4),
                                                        np.eye(6),
                                                        uncertain=True))

    for source_id in range(n_pcds):
        print('SOURCE: {}\nTARGET:'.format(source_id))
        for target_id in range(source_id + 1, n_pcds):
            print('\t{}'.format(target_id))
            if np.abs(target_id - source_id) > 2:
                # pose_graph.edges.append(
                #     o3d.pipelines.registration.PoseGraphEdge(source_id,
                #                                              target_id,
                #                                              np.eye(4),
                #                                              np.eye(6),
                #                                              uncertain=True))
                continue
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


def pairwise_registration_rgbd(source, target, intrinsic = np.eye(3)):
    color_info, hyrid_info = pairwise_registration_rgbd_aux(source, target, intrinsic)

    
    if hyrid_info[0]:
        transformation_icp = hyrid_info[1]
        information_icp = hyrid_info[2]

    else:
        if color_info[0]:
            transformation_icp = color_info[1]
            information_icp = color_info[2]
        else:
            transformation_icp = np.eye(4)
            information_icp = np.eye(6)

    return transformation_icp, information_icp


def pairwise_registration_rgbd_aux(source, target, imat = np.eye(3),
                            option = o3d.pipelines.odometry.OdometryOption()):
    phci = o3d.camera.PinholeCameraIntrinsic()
    phci.set_intrinsics(imat.height, imat.width, imat.fx, imat.fy, imat.ppx, imat.ppy)
    color_info = o3d.pipelines.odometry.compute_rgbd_odometry(
                source, target, phci, np.identity(4),
                o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(), option)
    hybrid_info = o3d.pipelines.odometry.compute_rgbd_odometry(
                source, target, phci, np.identity(4),
                o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
    return color_info, hybrid_info


def full_registration_rgbd(pcds, rgbds, intrinsic, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    print('RGBD')
    # Register all for loop closure:
    for source_id in range(n_pcds):
        for target_id in range(source_id+1, n_pcds):
            if source_id == target_id:
                continue
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                        target_id,
                                                        np.eye(4),
                                                        np.eye(6),
                                                        uncertain=True))

    for source_id in range(n_pcds):
        print('SOURCE: {}\nTARGET:'.format(source_id))
        for target_id in range(source_id + 1, n_pcds):
            print('\t{}'.format(target_id))
            if np.abs(target_id - source_id) > 2:
                # pose_graph.edges.append(
                #     o3d.pipelines.registration.PoseGraphEdge(source_id,
                #                                              target_id,
                #                                              np.eye(4),
                #                                              np.eye(6),
                #                                              uncertain=True))
                continue
            transformation_icp, information_icp = pairwise_registration_rgbd(
                rgbds[source_id], rgbds[target_id], intrinsic)

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
    if len(sys.argv) == 2:
        FRAME_ID = (np.arange(940)[::1]+1).tolist() #[352, 367, 382]
        #breakpoint()
        FRAME_ID = FRAME_ID[160:320]
        breakpoint()
    else:
        if len(sys.argv) == 3:
            FRAME_ID = [int(sys.argv[2])]
        elif len(sys.argv) == 4:
            frames = int(sys.argv[3]) - int(sys.argv[2]) + 1
            FRAME_ID = (np.arange(frames)[::1]+int(sys.argv[2])).tolist() #[352, 367, 382]
        elif len(sys.argv) == 5:
            frames = int(sys.argv[3]) - int(sys.argv[2]) + 1
            skip = int(sys.argv[4])
            FRAME_ID = (np.arange(frames)[::skip]+int(sys.argv[2])).tolist() #[352, 367, 382]
        else:
            print(len(sys.argv))
            sys.exit(1000)

        #FRAME_ID = [int(sys.argv[2])]
        img = cv2.imread( str(FILES_DIR / 'color/frame{:07d}.jpg'.format(FRAME_ID[0])) )
        print('type {}: shape {}'.format(type(img), img.shape))
        dimg = zarr.load( str(FILES_DIR / 'depth/frame{:07d}.zarr'.format(FRAME_ID[0]))) #, mode='r')
        print('type {}: shape {}'.format(type(dimg), dimg.shape))
        breakpoint()


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


    voxel_size = 0.02
    pcds_down = load_point_clouds(depth_intrinsic, FILES_DIR, FRAME_ID, voxel_size)
    rgbds = load_RGBDs(depth_intrinsic, FILES_DIR, FRAME_ID)

    print("Full registration ...")
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(pcds_down,
                                    max_correspondence_distance_coarse,
                                    max_correspondence_distance_fine)
        
        # pose_graph = full_registration_rgbd(pcds_down, rgbds, depth_intrinsic,
        #                             max_correspondence_distance_coarse,
        #                             max_correspondence_distance_fine)


    if True:
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
        print("Frame: {},\n{}".format(FRAME_ID[point_id],pose_graph.nodes[point_id].pose))
        pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
    o3d.visualization.draw_geometries(pcds_down,
                                    zoom=0.3412,
                                    front=[0.4257, -0.2125, -0.8795],
                                    lookat=[2.6172, 2.0475, 1.532],
                                    up=[-0.0694, -0.9768, 0.2024])


    o3d.io.write_pose_graph('test.json', pose_graph)


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
