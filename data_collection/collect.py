##########################################################
# A script to collect and save radar and camera data.    #
##########################################################

import os, sys, time
import multiprocessing as mp
from utils.ReadRadar import readradar 
from utils.ReadVideo import readvideo

"""
[The output format of radar points data]
    (1) .pkl file contains a list called: frameData
    (2) frameData[i]: dict -> dict(Data = detObj, Time = time.time(), Frame_ID = currentIndex)    # Frame_ID starts from 0
    (3) detObj: dict -> {"numObj": numDetectedObj, "x": x, "y": y, "z": z, "velocity": velocity}  # x, y, z, velocity: np array

[index]
    The index of radar data starts from 0
    The index of camera starts from 0
"""

if __name__ == '__main__':

    folderName = time.strftime('./data/%Y%m%d-%H%M%S')
    if not os.path.exists(folderName):
	    os.mkdir(folderName)

    dur = 20    # how many seconds
    video_fps = 20
    radar_fps = 20
    radar = readradar('./cfg/indoor.cfg', folderName, dur*radar_fps)
    
    # The initilization of radar is faster than the camera. Use pipe to synchronize.
    (pipe_send, pipe_receive) = mp.Pipe()
    
    mp.set_start_method(method = 'spawn')
    processes = []
    processes.append(mp.Process(target = readvideo, args = (pipe_receive, folderName, dur*video_fps, video_fps)))  
    processes.append(mp.Process(target = radar.run, args = (pipe_send, ) ))

    
    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()
