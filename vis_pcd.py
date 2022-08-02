from sys import argv
import numpy as np
import open3d as o3d
import sys
from pathlib import Path
from scipy.spatial.transform import Rotation
import zarr
from matplotlib import pyplot as plt
import cv2
import collections

import copy

DEPTH_SCALE = 0.001














# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 7x7 cross kernel
CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

# 7x7 diamond kernel
DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)


def fill_in_multiscale(depth_map, max_depth=100.0,
                       dilation_kernel_far=CROSS_KERNEL_3,
                       dilation_kernel_med=CROSS_KERNEL_5,
                       dilation_kernel_near=CROSS_KERNEL_7,
                       extrapolate=False,
                       blur_type='bilateral',
                       show_process=False):
    """Slower, multi-scale dilation version with additional noise removal that
    provides better qualitative results.
    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        dilation_kernel_far: dilation kernel to use for 30.0 < depths < 80.0 m
        dilation_kernel_med: dilation kernel to use for 15.0 < depths < 30.0 m
        dilation_kernel_near: dilation kernel to use for 0.1 < depths < 15.0 m
        extrapolate:whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'gaussian' - provides lower RMSE
            'bilateral' - preserves local structure (recommended)
        show_process: saves process images into an OrderedDict
    Returns:
        depth_map: dense depth map
        process_dict: OrderedDict of process images
    """

    # Convert to float32
    depths_in = np.float32(depth_map)

    # Calculate bin masks before inversion
    valid_pixels_near = (depths_in > 0.1) & (depths_in <= 15.0)
    valid_pixels_med = (depths_in > 15.0) & (depths_in <= 30.0)
    valid_pixels_far = (depths_in > 30.0)

    # Invert (and offset)
    s1_inverted_depths = np.copy(depths_in)
    valid_pixels = (s1_inverted_depths > 0.1)
    s1_inverted_depths[valid_pixels] = \
        max_depth - s1_inverted_depths[valid_pixels]

    # Multi-scale dilation
    dilated_far = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_far),
        dilation_kernel_far)
    dilated_med = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_med),
        dilation_kernel_med)
    dilated_near = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_near),
        dilation_kernel_near)

    # Find valid pixels for each binned dilation
    valid_pixels_near = (dilated_near > 0.1)
    valid_pixels_med = (dilated_med > 0.1)
    valid_pixels_far = (dilated_far > 0.1)

    # Combine dilated versions, starting farthest to nearest
    s2_dilated_depths = np.copy(s1_inverted_depths)
    s2_dilated_depths[valid_pixels_far] = dilated_far[valid_pixels_far]
    s2_dilated_depths[valid_pixels_med] = dilated_med[valid_pixels_med]
    s2_dilated_depths[valid_pixels_near] = dilated_near[valid_pixels_near]

    # Small hole closure
    s3_closed_depths = cv2.morphologyEx(
        s2_dilated_depths, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Median blur to remove outliers
    s4_blurred_depths = np.copy(s3_closed_depths)
    blurred = cv2.medianBlur(s3_closed_depths, 5)
    valid_pixels = (s3_closed_depths > 0.1)
    s4_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Calculate a top mask
    top_mask = np.ones(depths_in.shape, dtype=np.bool)
    for pixel_col_idx in range(s4_blurred_depths.shape[1]):
        pixel_col = s4_blurred_depths[:, pixel_col_idx]
        top_pixel_row = np.argmax(pixel_col > 0.1)
        top_mask[0:top_pixel_row, pixel_col_idx] = False

    # Get empty mask
    valid_pixels = (s4_blurred_depths > 0.1)
    empty_pixels = ~valid_pixels & top_mask

    # Hole fill
    dilated = cv2.dilate(s4_blurred_depths, FULL_KERNEL_9)
    s5_dilated_depths = np.copy(s4_blurred_depths)
    s5_dilated_depths[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image or create top mask
    s6_extended_depths = np.copy(s5_dilated_depths)
    top_mask = np.ones(s5_dilated_depths.shape, dtype=np.bool)

    top_row_pixels = np.argmax(s5_dilated_depths > 0.1, axis=0)
    top_pixel_values = s5_dilated_depths[top_row_pixels,
                                         range(s5_dilated_depths.shape[1])]

    for pixel_col_idx in range(s5_dilated_depths.shape[1]):
        if extrapolate:
            s6_extended_depths[0:top_row_pixels[pixel_col_idx],
                               pixel_col_idx] = top_pixel_values[pixel_col_idx]
        else:
            # Create top mask
            top_mask[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = False

    # Fill large holes with masked dilations
    s7_blurred_depths = np.copy(s6_extended_depths)
    for i in range(6):
        empty_pixels = (s7_blurred_depths < 0.1) & top_mask
        dilated = cv2.dilate(s7_blurred_depths, FULL_KERNEL_5)
        s7_blurred_depths[empty_pixels] = dilated[empty_pixels]

    # Median blur
    blurred = cv2.medianBlur(s7_blurred_depths, 5)
    valid_pixels = (s7_blurred_depths > 0.1) & top_mask
    s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    if blur_type == 'gaussian':
        # Gaussian blur
        blurred = cv2.GaussianBlur(s7_blurred_depths, (5, 5), 0)
        valid_pixels = (s7_blurred_depths > 0.1) & top_mask
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]
    elif blur_type == 'bilateral':
        # Bilateral blur
        blurred = cv2.bilateralFilter(s7_blurred_depths, 5, 0.5, 2.0)
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Invert (and offset)
    s8_inverted_depths = np.copy(s7_blurred_depths)
    valid_pixels = np.where(s8_inverted_depths > 0.1)
    s8_inverted_depths[valid_pixels] = \
        max_depth - s8_inverted_depths[valid_pixels]

    depths_out = s8_inverted_depths

    process_dict = None
    if show_process:
        process_dict = collections.OrderedDict()

        process_dict['s0_depths_in'] = depths_in

        process_dict['s1_inverted_depths'] = s1_inverted_depths
        process_dict['s2_dilated_depths'] = s2_dilated_depths
        process_dict['s3_closed_depths'] = s3_closed_depths
        process_dict['s4_blurred_depths'] = s4_blurred_depths
        process_dict['s5_combined_depths'] = s5_dilated_depths
        process_dict['s6_extended_depths'] = s6_extended_depths
        process_dict['s7_blurred_depths'] = s7_blurred_depths
        process_dict['s8_inverted_depths'] = s8_inverted_depths

        process_dict['s9_depths_out'] = depths_out

    return depths_out, process_dict















class intrinsics:
    # init method or constructor
    def __init__(self, model, width, height, fx, fy, ppx, ppy, coeffs = np.zeros(5)):
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
    #logging.debug('type {}: shape {}'.format(type(dimg), dimg.shape))
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

        dimg, _ = fill_in_multiscale(dimg)
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





def set_cam(C): 
        cam = o3d.geometry.LineSet.create_camera_visualization(view_width_px = intrinsic.width,
                                                                view_height_px = intrinsic.height, 
                                                                intrinsic = intrinsic.K, 
                                                                extrinsic = np.eye(4), 
                                                                scale=0.2)
        cam.transform(C)
        return cam

def set_cam_axis(C):
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    mesh_frame.transform(C)
    return mesh_frame



if __name__ == "__main__":


    depth_intrinsic = intrinsics('Brown Conrady', 1280,720, 636.773, 636.773, 636.435, 356.862 )
    color_intrinsic = intrinsics('Inverse Brown Conrady', 1280,720, 926.46, 926.289, 650.359, 359.765 )
    fake_intrinsic = intrinsics('Inverse Brown Conrady', 12, 8, 5, 5, 6, 4 )
    intrinsic = fake_intrinsic

    FILE_DIR = Path(sys.argv[1]) # MAIN PC
    FILE_DIR2 = Path(sys.argv[2]) # CEILING (REMOVE FOR VISUALIZATION)
    # TODO load stage in pieces via folder and keep in list

    ego_idx = 3
    EGO_DIR = Path(sys.argv[ego_idx]) # EGO FRAMES

    if len(sys.argv) == 4:
        FRAME_ID = (np.arange(940)[::1]+1).tolist() #[352, 367, 382]
        #breakpoint()
        FRAME_ID = FRAME_ID[160:320]
        breakpoint()
    else:
        if len(sys.argv) == 5:
            FRAME_ID = [int(sys.argv[ego_idx+1])]
        elif len(sys.argv) == 6:
            frames = int(sys.argv[ego_idx+2]) - int(sys.argv[ego_idx+1]) + 1
            FRAME_ID = (np.arange(frames)[::1]+int(sys.argv[ego_idx+1])).tolist() #[352, 367, 382]
        elif len(sys.argv) == 7:
            frames = int(sys.argv[ego_idx+2]) - int(sys.argv[ego_idx+1]) + 1
            skip = int(sys.argv[ego_idx+3])
            FRAME_ID = (np.arange(frames)[::skip]+int(sys.argv[ego_idx+1])).tolist() #[352, 367, 382]
        else:
            print(len(sys.argv))
            sys.exit(1000)



    #breakpoint()


    #breakpoint()
    print('Loading Point Clouds')
    pcds = load_point_clouds(depth_intrinsic, EGO_DIR, FRAME_ID, 0.05)
    #breakpoint()


    # hold camera extrinsics for rajectory
    I = np.eye(4)
    extrinsics = np.repeat(I[np.newaxis, :, :], len(pcds), axis=0)



    print("Loading main stage")
    pcd = o3d.io.read_point_cloud(str(FILE_DIR))
    pcd_ceiling = o3d.io.read_point_cloud(str(FILE_DIR2))


    print("Downsample the point cloud with a voxel of 0.05") # TODO Make a constant for config
    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    downpcd_ceiling = pcd_ceiling.voxel_down_sample(voxel_size=0.05)
    
    # TODO how does this affect ICP
    print("Recompute the normal of the downsampled point cloud")
    downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))




    # Initial Guess of Trajectory Start Pose
    R = Rotation.from_euler('y',165,degrees=True).as_matrix()
    C = np.eye(4)
    c = np.array([-1.2,1.7,-0.2])
    t =  - R @ c
    C[:3,:3] = R
    C[:3,3] = t

    basis_R = Rotation.from_euler('z',180,degrees=True).as_matrix()
    R =  R @ basis_R
    t = -R@c
    C[:3,:3] = R
    C[:3,3] = t


    cameras = []
    cam_axes = [] # camera axis  
    FRE_pcd = [downpcd + downpcd_ceiling] # combined stage
    prev_C = C
    print("Registering trajectory of len")
    for cam_idx in range(len(pcds)):

        # # For debug
        # noisy_C = np.copy(C)
        # noisy_C[:3,3] = noisy_C[:3,3] + 0
        #cam_init = set_cam(C) # previous guess for camera position
        
        
        print("[{}] Registering Frame".format(cam_idx))
        

        evaluation = o3d.pipelines.registration.evaluate_registration(
            pcds[cam_idx], downpcd, 0.05, prev_C)
        print("\tInitial alignment: {}".format(evaluation))
        # print("Transformation is:")
        # print(C)


        print("\tApplying point-to-point ICP...")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcds[cam_idx], downpcd, 0.5, prev_C,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        C = reg_p2p.transformation


        # print("\tTransformation is:")
        # print(reg_p2p.transformation)
        evaluation = o3d.pipelines.registration.evaluate_registration(
            pcds[cam_idx], downpcd, 0.5, C)

        print("\tFinal alignment: {}".format(evaluation))

        #draw_registration_result(source, target, reg_p2p.transformation)


        pcds[cam_idx].transform(C)
        cameras.append(set_cam(C))
        cam_axes.append(set_cam_axis(C))
        prev_C = C
        #breakpoint()




    
    
    o3d.visualization.RenderOption.line_width=8.0   
    #o3d.visualization.draw_geometries(FRE_pcd + [cam_init, mesh_frame] + [cam_result, mesh_frame_result] + pcds)
    o3d.visualization.draw_geometries(FRE_pcd + cameras + cam_axes + pcds)






    # print("Print a normal vector of the 0th point")
    # print(downpcd.normals[0])
    # print("Print the normal vectors of the first 10 points")
    # print(np.asarray(downpcd.normals)[:10, :])
    # print("")

    # print("Load a polygon volume and use it to crop the original point cloud")
    # vol = o3d.visualization.read_selection_polygon_volume(
    #     "../../TestData/Crop/cropped.json")
    # chair = vol.crop_point_cloud(pcd)
    # o3d.visualization.draw_geometries([chair])
    # print("")

    # print("Paint chair")
    # chair.paint_uniform_color([1, 0.706, 0])
    # o3d.visualization.draw_geometries([chair])
    # print("")