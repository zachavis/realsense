from pathlib import *
import cv2
import numpy as np
import zarr
import sys

#FILES_DIR = Path("test_videos/750f71d4-80ad-4a5f-bed4-c5d07ce69a64")

FILES_DIR = Path(sys.argv[1])
FRAME_ID = int(sys.argv[2])

img = cv2.imread( str(FILES_DIR / 'color/frame{:07d}.jpg'.format(FRAME_ID)) )
print('type {}: shape {}'.format(type(img), img.shape))
dimg = zarr.load( str(FILES_DIR / 'depth/frame{:07d}.zarr'.format(FRAME_ID))) #, mode='r')
print('type {}: shape {}'.format(type(dimg), dimg.shape))

color_image = img
depth_image = dimg.astype(np.float32)

#breakpoint()

# Apply colormap on depth image (image must be converted to 8-bit per pixel first)
depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs( depth_image, alpha=0.03), cv2.COLORMAP_JET)

depth_colormap_dim = depth_colormap.shape
color_colormap_dim = color_image.shape

# If depth and color resolutions are different, resize color image to match depth image for display
if depth_colormap_dim != color_colormap_dim:
    resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
    images = np.hstack((resized_color_image, depth_colormap))
else:
    images = np.hstack((color_image, depth_colormap))

# Show images
cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
cv2.imshow('RealSense', images)
cv2.waitKey(0)
print('done')
