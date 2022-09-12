from imp import C_BUILTIN
import os
from socket import CAN_BCM
import cv2

#import cv2.aruco
import numpy as np
import matplotlib.pyplot as plt

import fairotag as frt

import open3d as o3d

DATA_DIR = "data"
NUM_CALIB_SAMPLES = 5
MARKER_LENGTH = 0.094 # 0.05
MARKERS_WIDTH = 0.217
MARKERS_HEIGHT = 0.148

def get_test_reference():
    x = np.zeros(4)
    y = np.zeros(4)
    z = np.ones(4)
    rgb = np.zeros((4,3)).astype(np.float)
    rgb[0] = np.array([0,0,0])
    rgb[1] = np.array([1,0,0])
    rgb[2] = np.array([0,1,0])
    rgb[3] = np.array([0,0,1])

    x[0] = 0
    x[1] = 0
    x[2] = MARKERS_WIDTH
    x[3] = MARKERS_WIDTH

    y[0] = 0
    y[1] = MARKERS_HEIGHT
    y[2] = 0
    y[3] = MARKERS_HEIGHT

    xyz = np.stack((x,y,z),axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    # pcd.paint_uniform_color([1,0,0])
    return pcd

def set_cam(C, intrinsic): 
        cam = o3d.geometry.LineSet.create_camera_visualization(view_width_px = intrinsic.width,
                                                                view_height_px = intrinsic.height, 
                                                                intrinsic = intrinsic.K, 
                                                                extrinsic = np.eye(4), 
                                                                scale=0.2)
        cam.transform(C)
        return cam

def set_cam_axis(C, size=0.6):
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])
    mesh_frame.transform(C)
    return mesh_frame



def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


# points must have exact correspondance i.e. X[i] corresponds to Y[i] 
# def align_points (X, Y):

#     if X.shape[0] == 3:
#         A = X
#     else:
#         A = X.T

#     if Y.shape[0] == 3:
#         B = Y
#     else:
#         B = Y.T

#         breakpoint()

#     centroid_A = np.mean(A, axis=1)
#     centroid_B = np.mean(B, axis=1)

#     H = (A-centroid_A[:,None]) @ (B-centroid_B[:,None]).T # from A to B
#     U, S, Vh = np.linalg.svd(H)

#     R_A2B = Vh.T @ U.T

#     if np.linalg.det(R_A2B) < 0:
#         U,S,Vh = np.linalg.svd(R_A2B)
#         Vh[-1] *= -1
#         R_A2B = Vh.T @ U.T

#     t_A2B = centroid_B - R_A2B @ centroid_A

#     C = np.eye(4)
#     #C[:3,:3] = R_A2B
#     C[:3,3] = t_A2B

#     return C

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

depth_intrinsic = intrinsics('Brown Conrady', 1280, 720, 636.773, 636.773, 636.435, 356.862 )
color_intrinsic = intrinsics('Inverse Brown Conrady', 1280, 720, 926.46, 926.289, 650.359, 359.765 )
#breakpoint()

img = cv2.imread(os.path.join(DATA_DIR, "test_Color.png"))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# Initialize module
cam = frt.CameraModule()
cam.set_intrinsics(color_intrinsic)

# breakpoint()
# print('test')

# # Calibrate camera
# # (Images of the Charuco board taken at different angles are labeled as "charuco_<number>.jpg".)
# imgs = []
# for i in range(NUM_CALIB_SAMPLES):
#     filename = os.path.join(DATA_DIR, f"charuco_{i + 1}.jpg")
#     img_calib = cv2.imread(filename)
#     imgs.append(img_calib)

# c.calibrate_camera(imgs)

# # Register markers
# c.register_marker_size(0, MARKER_LENGTH)
# c.register_marker_size(3, MARKER_LENGTH)
# c.register_marker_size(4, MARKER_LENGTH)

cam.register_marker_size(4, MARKER_LENGTH)
cam.register_marker_size(5, MARKER_LENGTH)
cam.register_marker_size(6, MARKER_LENGTH)
cam.register_marker_size(7, MARKER_LENGTH)

markers = cam.detect_markers(img) # TODO: Handle < 4 markers being seen
print(markers)
#breakpoint()

img_rend = cam.render_markers(img, markers=markers)
plt.imshow(cv2.cvtColor(img_rend, cv2.COLOR_BGR2RGB))
plt.show()


pcd = get_test_reference()
point_reference = np.asarray(pcd.points)
point_query = np.array([m.pose.matrix()[:3,3] for m in markers if m.pose is not None])

# centroid_reference = np.mean(point_reference, axis=0)
# centroid_query = np.mean(point_query, axis=0)

# centered_reference = point_reference - centroid_reference
# centered_query = point_query - centroid_query

# H = centered_query.T @ centered_reference # from query to reference
# U, S, Vh = np.linalg.svd(H)

# R_query_ref = Vh.T @ U.T

# if np.linalg.det(R_query_ref) < 0:
#     U,S,Vh = np.linalg.svd(R_query_ref)
#     Vh[-1] *= -1
#     R_query_ref = Vh.T @ U.T

# t_query_ref = centroid_reference - R_query_ref @ centroid_query

# C_query_ref = np.eye(4)
# C_query_ref[:3,:3] = R_query_ref
# C_query_ref[:3,3] = t_query_ref

R, t = rigid_transform_3D(point_query.T, point_reference.T)
breakpoint()
C_query_ref = np.eye(4)
C_query_ref[:3,:3] = R
C_query_ref[:3,3] = t[:,0]

markers3d_id = [m.id for m in markers if m.pose is not None]
correctOrder = np.argsort(markers3d_id)
markers = [markers[i] for i in correctOrder]

markers3d = []
markers3d = [set_cam_axis(m.pose.matrix(), MARKER_LENGTH) for m in markers if m.pose is not None]
# breakpoint()
cam3d = set_cam(np.eye(4),intrinsics('Inverse Brown Conrady', 12, 8, 5, 5, 6, 4 ))

# breakpoint()
# markers3d = [markers3d[i] for i in correctOrder]
# markers3d_id = [markers3d_id[i] for i in correctOrder]

breakpoint()

markers3d_new = []
markers3d_new = [set_cam_axis(C_query_ref @ m.pose.matrix(), MARKER_LENGTH) for m in markers if m.pose is not None]
cam3d_new = set_cam(C_query_ref,intrinsics('Inverse Brown Conrady', 12, 8, 5, 5, 6, 4 ))

o3d.visualization.draw_geometries( [pcd] + markers3d  + [cam3d])
o3d.visualization.draw_geometries( [pcd] + markers3d_new + [cam3d_new])
o3d.visualization.draw_geometries( [pcd] + markers3d + markers3d_new + [cam3d, cam3d_new])