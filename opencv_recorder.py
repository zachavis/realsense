## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

# https://blog.csdn.net/qq_43297710/article/details/121578249
# https://qiita.com/yumion/items/6eeb820c1f06839d57a7

import pyrealsense2 as rs
import numpy as np
import cv2
import zarr
import lzma
lzma_filters = [dict(id=lzma.FILTER_DELTA, dist=4), dict(id=lzma.FILTER_LZMA2, preset=1)]
from numcodecs import LZMA
compressor = LZMA(filters=lzma_filters)
import uuid

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

fourcc = cv2.VideoWriter_fourcc(*'XVID') #'h264')
#out_depth = cv2.VideoWriter("videos/depth/depth_vid.avi", fourcc, 30, (640, 480), 0)
out_color = cv2.VideoWriter("videos/color_vid.avi", fourcc, 30, (640, 480), True)

i = -1
depth_images = np.zeros((300, 480, 640)) # 10 second chunks at 30fps # DON'T FORGET TO ADD ZERO FRAME
try:
    while True:
        i += 1

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            # write previous frame RGB
            # write previous frame DEPTH
            #if prev_depth_frame and prev_color_frame:
            	#out.write(
            continue
        prev_depth_frame = depth_frame
        prev_color_frame = color_frame

        # Convert images to numpy arrays
        depth_image = np.expand_dims(np.asanyarray(depth_frame.get_data()),-1)
        #cv2.imwrite("videos/depth/frame{:07d}.".format(i), np.uint8(255 * depth_image))
        #skimage.io.imsave("videos/depth/frame{:07d}.".format(i), depth_image, plugin="tifffile")
        #np.save("videos/depth/frame{:07d}".format(i), depth_image)
        #z = zarr.array(depth_image, chunks=(80, 80), compressor=compressor)
        if i + 1 == 300: 
        	zarr.save("videos/depth/frame{:07d}".format(i), depth_images)
        	depth_images *= 0
        else:
        	depth_images[i] = depth_image[:,:,0]
        
        color_image = np.asanyarray(color_frame.get_data())
        
        #breakpoint()
        out_color.write(color_image)
        #out_depth.write(np.uint8(255 * depth_image))

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

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
        #cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q') or i == 300:
            break

finally:

    # Stop streaming
    pipeline.stop()
