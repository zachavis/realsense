## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

# https://blog.csdn.net/qq_43297710/article/details/121578249
# https://qiita.com/yumion/items/6eeb820c1f06839d57a7

# RealSense Camera
import pyrealsense2 as rs

# Vision
import numpy as np
import cv2

# Compression
import zarr
import lzma
lzma_filters = [dict(id=lzma.FILTER_DELTA, dist=4), dict(id=lzma.FILTER_LZMA2, preset=1)]
from numcodecs import LZMA, Blosc
delta_compressor = LZMA(filters=lzma_filters)
zstd_compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.SHUFFLE)

# IDs
import uuid

# Audio
import pyaudio
import wave

# System
import threading
import subprocess
import time
import os
from pathlib import * 
import sys


# Input
from pynput.mouse import Listener as MouseListener
from pynput.keyboard import Listener as KeyboardListener
from pynput.keyboard import KeyCode
from pynput import keyboard, mouse



RECORDING = False
BEGIN_RECORDING = False
QUIT_RECORDING = False
KILL_PROG = False
KEY_PRESS_SAFETY = False
PRE_RECORD_BEEP = True



VIDEO_DIR = Path("test_videos")





#######################################################################
#### INPUT CALLBACKS ##############################################
###############################################################

def on_press(key):
    global RECORDING
    global BEGIN_RECORDING
    global QUIT_RECORDING
    global KEY_PRESS_SAFETY
    global KILL_PROG
    #print("Key pressed: {0}".format(key))
    if not KEY_PRESS_SAFETY:
        #print(type(key))
        #print('key is', key)
        if key == keyboard.Key.esc or key  == KeyCode.from_char('q'):
            RECORDING = False
            KILL_PROG = True
            #print('kill')
        else:
            RECORDING = not RECORDING
            KEY_PRESS_SAFETY = True
    
def on_release(key):
    global RECORDING
    global BEGIN_RECORDING
    global QUIT_RECORDING
    global KEY_PRESS_SAFETY
    #print("Key pressed: {0}".format(key))
    KEY_PRESS_SAFETY = False
   
def on_click(x, y, button, pressed):
    global RECORDING
    global BEGIN_RECORDING
    global QUIT_RECORDING
    global KEY_PRESS_SAFETY
    global KILL_PROG
    if pressed and not KEY_PRESS_SAFETY:
        #print("Click pressed: {0}".format(button))
        #if not KEY_PRESS_SAFETY:
        if button == mouse.Button.middle:
            RECORDING = False
            KILL_PROG = True
            #print('kill')
        else:
            RECORDING = not RECORDING
            KEY_PRESS_SAFETY = True
    else:
        KEY_PRESS_SAFETY = False




#######################################################################
#### STREAMING DEVICES ############################################
###############################################################

audio_thread = None

class AudioRecorder():


    # Audio class based on pyAudio and Wave
    def __init__(self, filename = "temp_audio.wav"):

        self.open = True
        self.rate = 44100
        self.frames_per_buffer = 1024
        self.channels = 2
        self.format = pyaudio.paInt16
        self.audio_filename = filename #"temp_audio.wav"
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer = self.frames_per_buffer)
        self.audio_frames = []


    # Audio starts being recorded
    def record(self):

        self.stream.start_stream()
        while(self.open == True):
            data = self.stream.read(self.frames_per_buffer) 
            self.audio_frames.append(data)
            if self.open==False:
                break


    # Finishes the audio recording therefore the thread too    
    def stop(self):

        if self.open==True:
            self.open = False
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()

            waveFile = wave.open(self.audio_filename, 'wb')
            waveFile.setnchannels(self.channels)
            waveFile.setsampwidth(self.audio.get_sample_size(self.format))
            waveFile.setframerate(self.rate)
            waveFile.writeframes(b''.join(self.audio_frames))
            waveFile.close()

        pass

    # Launches the audio recording function using a thread
    def start(self):
        audio_thread = threading.Thread(target=self.record)
        audio_thread.start()






class VideoRecorder():  

    # Video class based on openCV 
    def __init__(self):

        self.open = True
        self.device_index = 0
        self.fps = 6               # fps should be the minimum constant rate at which the camera can
        self.fourcc = "MJPG"       # capture images (with no decrease in speed over time; testing is required)
        self.frameSize = (640,480) # video formats and sizes also depend and vary according to the camera used
        self.video_filename = "temp_video.avi"
        self.video_cap = cv2.VideoCapture(self.device_index)
        self.video_writer = cv2.VideoWriter_fourcc(*self.fourcc)
        self.video_out = cv2.VideoWriter(self.video_filename, self.video_writer, self.fps, self.frameSize)
        self.frame_counts = 1
        self.start_time = time.time()


    # Video starts being recorded 
    def record(self):

#       counter = 1
        timer_start = time.time()
        timer_current = 0


        while(self.open==True):
            ret, video_frame = self.video_cap.read()
            if (ret==True):

                    self.video_out.write(video_frame)
#                   print str(counter) + " " + str(self.frame_counts) + " frames written " + str(timer_current)
                    self.frame_counts += 1
#                   counter += 1
#                   timer_current = time.time() - timer_start
                    time.sleep(0.16)
#                   gray = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
#                   cv2.imshow('video_frame', gray)
#                   cv2.waitKey(1)
            else:
                break

                # 0.16 delay -> 6 fps
                # 


    # Finishes the video recording therefore the thread too
    def stop(self):

        if self.open==True:

            self.open=False
            self.video_out.release()
            self.video_cap.release()
            cv2.destroyAllWindows()

        else: 
            pass


    # Launches the video recording function using a thread          
    def start(self):
        video_thread = threading.Thread(target=self.record)
        video_thread.start()




def start_AVrecording(filename):

    global video_thread
    global audio_thread

    video_thread = VideoRecorder()
    audio_thread = AudioRecorder()

    audio_thread.start()
    video_thread.start()

    return filename



def start_video_recording(filename):

    global video_thread

    video_thread = VideoRecorder()
    video_thread.start()

    return filename


def start_audio_recording(filename):

    global audio_thread

    audio_thread = AudioRecorder(filename)
    audio_thread.start()

    return filename


def stop_audio_recording(filename = ''):
    global audio_thread
    
    audio_thread.stop() 
    #frame_counts = video_thread.frame_counts
    #elapsed_time = time.time() - video_thread.start_time
    #recorded_fps = frame_counts / elapsed_time
    #print "total frames " + str(frame_counts)
    #print "elapsed time " + str(elapsed_time)
    #print "recorded fps " + str(recorded_fps)
    #video_thread.stop() 

    # Makes sure the threads have finished
    while threading.active_count() > 1:
        time.sleep(1)



def stop_AVrecording(filename):

    audio_thread.stop() 
    frame_counts = video_thread.frame_counts
    elapsed_time = time.time() - video_thread.start_time
    recorded_fps = frame_counts / elapsed_time
    print("total frames " + str(frame_counts))
    print("elapsed time " + str(elapsed_time))
    print("recorded fps " + str(recorded_fps))
    video_thread.stop() 

    # Makes sure the threads have finished
    while threading.active_count() > 1:
        time.sleep(1)


#    Merging audio and video signal

    if abs(recorded_fps - 6) >= 0.01:    # If the fps rate was higher/lower than expected, re-encode it to the expected

        print("Re-encoding")
        cmd = "ffmpeg -r " + str(recorded_fps) + " -i temp_video.avi -pix_fmt yuv420p -r 6 temp_video2.avi"
        subprocess.call(cmd, shell=True)

        print("Muxing")
        cmd = "ffmpeg -ac 2 -channel_layout stereo -i temp_audio.wav -i temp_video2.avi -pix_fmt yuv420p " + filename + ".avi"
        subprocess.call(cmd, shell=True)

    else:

        print("Normal recording\nMuxing")
        cmd = "ffmpeg -ac 2 -channel_layout stereo -i temp_audio.wav -i temp_video.avi -pix_fmt yuv420p " + filename + ".avi"
        subprocess.call(cmd, shell=True)

        print("..")





# Required and wanted processing of final files
def file_manager(filename):

    local_path = os.getcwd()

    if os.path.exists(str(local_path) + "/temp_audio.wav"):
        os.remove(str(local_path) + "/temp_audio.wav")

    if os.path.exists(str(local_path) + "/temp_video.avi"):
        os.remove(str(local_path) + "/temp_video.avi")

    if os.path.exists(str(local_path) + "/temp_video2.avi"):
        os.remove(str(local_path) + "/temp_video2.avi")

    if os.path.exists(str(local_path) + "/" + filename + ".avi"):
        os.remove(str(local_path) + "/" + filename + ".avi")








class RealSenseRecorder():  

    # Video class based on openCV 
    def __init__(self):

        self.open = True
        self.device_index = 0
        self.fps = 6               # fps should be the minimum constant rate at which the camera can
        self.fourcc = "MJPG"       # capture images (with no decrease in speed over time; testing is required)
        self.frameSize = (640,480) # video formats and sizes also depend and vary according to the camera used
        self.video_filename = "temp_video.avi"
        self.video_cap = cv2.VideoCapture(self.device_index)
        self.video_writer = cv2.VideoWriter_fourcc(*self.fourcc)
        self.video_out = cv2.VideoWriter(self.video_filename, self.video_writer, self.fps, self.frameSize)
        self.frame_counts = 1
        self.start_time = time.time()


    # Video starts being recorded 
    def record(self):

#       counter = 1
        timer_start = time.time()
        timer_current = 0


        while(self.open==True):
            ret, video_frame = self.video_cap.read()
            if (ret==True):

                    self.video_out.write(video_frame)
#                   print str(counter) + " " + str(self.frame_counts) + " frames written " + str(timer_current)
                    self.frame_counts += 1
#                   counter += 1
#                   timer_current = time.time() - timer_start
                    time.sleep(0.16)
#                   gray = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
#                   cv2.imshow('video_frame', gray)
#                   cv2.waitKey(1)
            else:
                break

                # 0.16 delay -> 6 fps
                # 


    # Finishes the video recording therefore the thread too
    def stop(self):

        if self.open==True:

            self.open=False
            self.video_out.release()
            self.video_cap.release()
            cv2.destroyAllWindows()

        else: 
            pass


    # Launches the video recording function using a thread          
    def start(self):
        video_thread = threading.Thread(target=self.record)
        video_thread.start()




DEPTH_WIDTH = 1280
DEPTH_HEIGHT = 720
COLOR_WIDTH = DEPTH_WIDTH #1920
COLOR_HEIGHT = DEPTH_HEIGHT #1080

#######################################################################
#### MAIN EXECUTION ###############################################
###############################################################
if __name__ == "__main__":


    #print('cmd entry:', sys.argv)
    VIDEO_DIR = sys.argv[1]

    # Setup the listener threads
    keyboard_listener = KeyboardListener(on_press=on_press, on_release=on_release)
    mouse_listener = MouseListener(on_click=on_click)

    keyboard_listener.start()
    mouse_listener.start()

    while not KILL_PROG: # Configure streams, folder paths, and save video

        # Standby phase
        print('Press any key to begin recording [Esc, q, or middle click to Quit]',end='',flush=True)
        while not RECORDING and not KILL_PROG:
            if PRE_RECORD_BEEP:
                print('\a.',end='',flush=True)
            else:
                print('.', end='', flush=True)
            time.sleep(0.5)
            
        print('')
        if KILL_PROG:
            break
            
        # Generate Unique Video ID and corresponding folder
        video_name = uuid.uuid4()
        print('Recording {}'.format(video_name))
        uuid_dir = Path(str(video_name))
        (VIDEO_DIR / uuid_dir).mkdir(parents=True, exist_ok=True)
        (VIDEO_DIR / uuid_dir / Path('color')).mkdir(parents=True, exist_ok=True)
        (VIDEO_DIR / uuid_dir / Path('depth')).mkdir(parents=True, exist_ok=True)
	    
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

        #config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        #if device_product_line == 'L500':
        #    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        #else:
        #    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, 30)

        # Start streaming
        pipeline.start(config)

        fourcc = cv2.VideoWriter_fourcc(*'XVID') #'h264')
        #out_depth = cv2.VideoWriter("videos/depth/depth_vid.avi", fourcc, 30, (640, 480), 0)
        out_color = cv2.VideoWriter( str(VIDEO_DIR / uuid_dir / Path("color_vid.avi")), fourcc, 30, (COLOR_WIDTH, COLOR_HEIGHT), True)

        i = 0
        #zarr_depth_images = zarr.zeros((300,480,640), chunks=(100,100,100), dtype=np.float16, compressor=zstd_compressor)
        zarr_depth_frame = zarr.zeros((DEPTH_HEIGHT,DEPTH_WIDTH), dtype=np.float16, compressor=zstd_compressor)
        #depth_images = np.zeros((300, 480, 640)) # 10 second chunks at 30fps # DON'T FORGET TO ADD ZERO FRAME
        try:
            #start_audio_recording( str(VIDEO_DIR / uuid_dir / 'test.wav') )
            while RECORDING:
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
                zarr_depth_frame[:,:] = depth_image[:,:,0]
                #cv2.imwrite("videos/depth/frame{:07d}.".format(i), np.uint8(255 * depth_image))
                #skimage.io.imsave("videos/depth/frame{:07d}.".format(i), depth_image, plugin="tifffile")
                #np.save("videos/depth/frame{:07d}".format(i), depth_image)
                #z = zarr.array(depth_image, chunks=(80, 80), compressor=compressor)
                
                
                #if i + 1 == 300: 
                #	zarr.save( str(VIDEO_DIR / uuid_dir / Path('depth/frame{:07d}'.format(i))), zarr_depth_images)
                #	zarr_depth_images[:,:,:] = 0
                #else:
                #	zarr_depth_images[i] = depth_image[:,:,0]
                #
                color_image = np.asanyarray(color_frame.get_data())
                
                #breakpoint()
                out_color.write(color_image)
                
                cv2.imwrite( str(VIDEO_DIR / uuid_dir / Path('color/frame{:07d}.jpg'.format(i))), color_image)
                
                color_image = cv2.resize(color_image, (DEPTH_WIDTH, DEPTH_HEIGHT)) #, interpolation = cv2.INTER_AREA)
                
                zarr.save( str(VIDEO_DIR / uuid_dir / Path('depth/frame{:07d}.zarr'.format(i))), zarr_depth_frame)
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
                cv2.waitKey(1)
                #if cv2.waitKey(1) & 0xFF == ord('q') or i == 300:
                #    break


        finally:

	        # Stop streaming
            pipeline.stop()
            #stop_audio_recording()
            print('Video saved')
            
    print('Program terminated...')
