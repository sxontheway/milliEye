########################################################################
# A script to run radar and camera and show the results in real-time.  #
########################################################################

import cv2
import numpy as np
import time
import os
import multiprocessing as mp
from collections import deque
from show import *
from utils.ReadRadar import *
from utils.tracking import *
from utils.utils import *


def image_generation(q: mp.Queue(), camera_fps):

    # Camera Initialization
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    cap.set(5, camera_fps)

    while True:
        try:
            ret, frame = cap.read()
            if ret:
                q.put(frame)
                q.get() if q.qsize() > 1 else time.sleep(0.001)
        except:
            cap.release()


def radar_processing(q: mp.Queue(), radar: readradar(), params: list):

    radar_fps, max_depth, num_pts_filter, velocity_thres, dbscan_weights, show_info = params

    # Configurate the serial port
    CLIport, Dataport = {}, {}
    CLIport, Dataport = serialConfig(radar.configFileName)

    # Get the configuration parameters from the configuration file
    configParameters = parseConfigFile(radar.configFileName)

    dataOk = 0
    detObj = {}

    frame = q.get()
    loop_fps = 0

    cv2.namedWindow("fusion", 0)
    cv2.resizeWindow("fusion", 960, 720)
    font = cv2.FONT_HERSHEY_SIMPLEX
    calib_param = load_calib("./yaml/calib_FOV90.yaml")

    # "(3,)<f4" means little-endian; an array with size (3, ); each element is a 4 bytes floats
    dtype_clusters = np.dtype({'names': ('num_points', 'center', 'size', 'avgV'),
                               'formats': ['<u4', '(3,)<f4', '(3,)<f4', '<f4']})
    tracker = Tracker(dtype_clusters, radar_fps)

    while True:
        try:
            time_start = time.time()

            # read radar frames
            dataOk, frameNumber, detObj, timestamp = radar.readAndParseData68xx(
                Dataport, configParameters)

            # check if there is data
            if dataOk:
                """
                q.get() is a block operation. It will slow down the radar while loop, 
                so that the `readAndParseData68xx` cannot keep up with the speed of radar sensor (20fps or 30fps), 
                which will cause a delayed plotting of point cloud.
                """
                if q.qsize() > 0:   # A block operation. Wait untill another process send the captured image.
                    frame = q.get()

                # show FPS
                cv2.putText(frame, "Radar Loop FPS: {:.1f}".format(loop_fps),
                            (5, 15), font, 0.5, (255, 0, 0), 2)

                # projection
                x, y, z, v = detObj["x"], detObj["y"], detObj["z"], detObj["velocity"]
                points_3d = np.array([x, y, z, v])
                uv, xyzV = from_3d_to_2d(points_3d, calib_param)   # 2d points in image coordinate

                # FOV and velocity filter
                filtering = [0<=i[0]<640 and 0<=i[1]<480 and j[2]<max_depth and abs(j[3])>= velocity_thres for i, j in zip(uv, xyzV)]
                uv, xyzV = uv[filtering], xyzV[filtering]

                # clustering and tracking
                clusters, labels = radar_dbscan(xyzV, dtype_clusters, dbscan_weights)
                clusters = clusters[clusters["num_points"] >= num_pts_filter]
                tracked_clusters = tracker.update(clusters)

                # Tracking in 3D and draw 3d bboxes on image
                for cluster in tracked_clusters:   
                    center = cluster['center']  
                    size = cluster['size']
                    if max(size) < 3:
                        draw_3d_boxes(center, size, frame, calib_param)
                    tmp1, tmp2 = projection_xyr_to_uv(center, calib_param)
                    cv2.circle(frame, (int(tmp1), int(tmp2)), 7 , (255,255,255), -1)
                    print(center, size, cluster['avgV'])

                # Showed clustered points and draw bboxes using merely clustered 2D points
                """
                for i in range(len(clusters)):
                    for xy in uv[labels == i]:
                        cv2.circle(frame, (xy[0], xy[1]), 7, (0, 255, 255), -1)
                for i in range(len(clusters)):    
                    tmp = uv[labels == i]
                    cv2.rectangle(frame, (min(tmp[:, 0]), min(tmp[:, 1])), (max(tmp[:, 0]), max(tmp[:, 1])), (255,0,255), 1) 
                """

                # show points
                for i, j in zip(uv, xyzV):
                    color = (0, int((1-j[2]/max_depth)*255), int(j[2]/max_depth*255))     # colors change according to the depth
                    cv2.circle(frame, (i[0], i[1]), 7, color, -1)
                    if show_info:
                        cv2.putText(frame, "z={:.2f}".format(xyzV[2]), \
                            (uv[0], uv[1]), font, 0.5, (0, 255, 255), 1)
                        cv2.putText(frame, "v={:.2f}".format(xyzV[3]), \
                            (uv[0], uv[1]+15), font, 0.5, (0, 255, 255), 1)

                cv2.imshow("fusion", frame)
                cv2.waitKey(1)

                # calculate used time
                time_end = time.time()
                loop_fps = 1/(time_end-time_start)

        except KeyboardInterrupt:
            CLIport.write(('sensorStop\n').encode())
            CLIport.close()
            Dataport.close()
            cv2.destroyAllWindows()


if __name__ == '__main__':

    # params for camera
    camera_fps = 20

    # params for radar
    radar_fps = 20          # filter out points with larger depth than the threshold
    max_depth = 15
    num_pts_filter = 3      # filter out clusters with fewer number of points
    velocity_thres = 0.2    # filter out points with smaller velocity than the threshold
    dbscan_weights = [1,1,3,1]        # weights of metrics using in DBScan, in the order of directions uvzv
    show_info = 0                      # whether show distance and velocity in the figure

    radar = readradar('./cfg/indoor.cfg')      # A class to read radar data

    mp.set_start_method(method='spawn', force=True)
    queue = mp.Queue(maxsize=4)

    processes = []
    processes.append(mp.Process(
        target=image_generation, args=(queue, camera_fps)))
    processes.append(mp.Process(target=radar_processing, args=(queue, radar, \
        [radar_fps, max_depth, num_pts_filter, velocity_thres, dbscan_weights, show_info])))

    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()


""" The cv2.VideoCapture.get() parameterss
See: https://blog.csdn.net/u011436429/article/details/80604590
CAP_PROP_EXPOSURE 	Exposure Time
-1 	       		    640 ms
-2 			        320 ms
-3 			        160 ms
-4 			        80 ms
-5 			        40 ms
-6 			        20 ms
-7 			        10 ms
-8 			        5 ms
-9 			        2.5 ms
-10 			    1.25 ms
-11 			    650 µs
-12 			    312 µs
-13 			    150 µs
"""


# The single process one is in the following.
# The reading of mmWave radar points cloud will suffer severe delay.
""" 
if __name__ == '__main__':

    # Camera Initialization
    video_fps = 30
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)	
    cap.set(4, 480)
    cap.set(5, video_fps)

    # Radar Initialization
    radar = readradar('./cfg/indoor.cfg')

    # Configurate the serial port
    CLIport, Dataport = {}, {} 
    CLIport, Dataport = serialConfig(radar.configFileName)

    # Get the configuration parameters from the configuration file
    configParameters = parseConfigFile(radar.configFileName)

    detObj = {}  
    points_buffer = deque()   

    while True:
        try:

            time_start=time.time()

            # read video frames, it is a block operation
            ret, frame = cap.read()

            # read radar frames
            dataOk = 0

            dataOk, frameNumber, detObj, timestamp = radar.readAndParseData68xx(Dataport, configParameters)
            print(dataOk)

            # check if there is data
            if dataOk:
                x, y, z = detObj["x"], detObj["y"], detObj["z"]
                points_3d = np.array([x, y, z])

                # projection
                calib_param = load_calib("./yaml/calibration.yaml")
                points_2d = projection(points_3d, calib_param)

                # show
                for i in points_2d:
                    cv2.circle(frame, (i[0], i[1]), 7, (0,0,255), -1)
                cv2.imshow("capture", frame)

            time_end = time.time()
            print('time cost', time_end-time_start,'s') 

            if  cv2.waitKey(1) & 0xFF == ord('p'):
                cv2.waitKey(10000)

        except KeyboardInterrupt:
            CLIport.write(('sensorStop\n').encode())
            CLIport.close()
            Dataport.close()
            cap.release()
            cv2.destroyAllWindows()
            break
"""
