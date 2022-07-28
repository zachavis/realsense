import numpy as np
import open3d as o3d

# potentially helpful for doing rosbag stuff
# https://github.com/IntelRealSense/librealsense/issues/4934#issuecomment-537705225

# read depth image


# multiply by scale for distance
# https://drive.google.com/file/d/1z8i1wjD946RBh84NhxTIfJt5jGRxhMLr/view?usp=sharing
depth_scale = 0.001

# intrinsic params
# https://answers.ros.org/question/363236/intel-realsense-d435-intrinsic-parameters/
# https://github.com/IntelRealSense/realsense-ros/issues/750
# test = {
#     D: [0.0, 0.0, 0.0, 0.0, 0.0],
#     K: [639.0849609375, 0.0, 644.4653930664062, 0.0, 639.0849609375, 404.2340393066406, 0.0, 0.0, 1.0],
#     R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
#     P: [639.0849609375, 0.0, 644.4653930664062, 0.0, 0.0, 639.0849609375, 404.2340393066406, 0.0, 0.0, 0.0, 1.0, 0.0]
# }


class camera_intrinsics:
    # for 640x360 camera
    fx = 639.0849609375 #322.282410
    fy = 639.0849609375 #322.282410
    ppx = 1280 / 2 # 320.818268
    ppy = 720 / 2 # 178.779297

# https://github.com/IntelRealSense/realsense-ros/issues/551

def convert_depth_frame_to_pointcloud(depth_image, camera_intrinsics, depth_scale = 0.001 ):
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

	return x, y, z


if __name__ == '__main__':
    print('Hello, world!')