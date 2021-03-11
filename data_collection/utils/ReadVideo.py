import cv2
import sys, os, time
import numpy as np

def readvideo(pipe, folderName, num, fps):

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    cap.set(5, fps)

    video_inf = dict(weight = int(cap.get(3)), height = int(cap.get(4)), fps = int(cap.get(5)))
    print(video_inf) 

    timestampsFile = folderName + '/timestamps' + '.txt'
    videoFile = folderName + '/video' + '.mp4'

    f = open(timestampsFile, 'w')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vout_1 = cv2.VideoWriter()
    vout_1.open(videoFile, fourcc, fps, (video_inf["weight"], video_inf["height"]), True)

    count = 0

    # Synchronize two sensors
    cap.read()   # a warming up, becasue the first time of `cap.read()` takes long time (around ~0.8s)
    pipe.recv()     # wait radar to be prepared
    pipe.send("Camera is ready")
    print("Camera -> Radar: camera is ready to start")

    while(True):
        ret, frame = cap.read()

        if ret:
            f.write("{} {}\n".format(time.time(), count))
            print("Video count: " + str(count))
            vout_1.write(frame)

        if  count == num or cv2.waitKey(1) & 0xFF==ord("w"):
            print("Camera Done!")
            break
        count += 1

    vout_1.release()
    cap.release()
    cv2.destroyAllWindows()
    f.close()
