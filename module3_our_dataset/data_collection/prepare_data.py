#########################################################
# A script to show collected data / generate dataset.	#
# show = 0: generate dataset ; show = 1: show dataset	#
#########################################################


import numpy as np
import os
import cv2
import re
import pickle
from utils.utils import *


def xywh2xyxy(x):
    y = np.empty(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


if __name__ == "__main__":

    # show = 0: generate dataset ; show = 1: show dataset
    show = 0

    # radar-related params
    fps_divider = 5                 # the denominator of fps
    radar_fps = 20                  # the fps of radar data                         
    num_nearest = 3                 # for each video frame, find "num_nearest" radar frames 
    overlay_num = 2                 # the point cloud from "overlay_num" radar frames are aggregared together
    
    dbscan_weights = [2,1,3,1]      # weights of xyzV
    dbscan_eps = 1.5                # the hyper-param in DBSCAN
    
    num_pts_filter = 5              # remove clusters whose points numbers are below the threshold 
    min_velocity = 0.1              # the velocity threshold of points
    max_size = 20                   # the upper bound of the proposed 3d box sizes
    max_depth = 50                  # the max depth of points 
    
    max_age=4                       # keep tracking for how many frames                 
    min_hits=4                      # start tracking for how many frames
    calib_param = load_calib("./yaml/calib_FOV90.yaml")     # calibration matrix


    # create folders for the dataset
    folders = ["./dataset/image", "./dataset/label", "./dataset/radar_box", "./dataset/radar_point"]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
    f = open("./dataset/dataset.txt", 'w'); f.close()


    # iterate every folder
    raw_data_folder = "./data/"
    folder_base_list = sorted([string for string in os.listdir(raw_data_folder) if re.match(r'(.*)-(.*)', string)])
    folder_idxs = range(len(folder_base_list))[-1:]

    for folder_idx in folder_idxs:
        folder_base = raw_data_folder + folder_base_list[folder_idx]

        # do synchronization by matching timestamps
        video_stamps, point_data = load_data(           # video_stamps starts from 0
            os.path.join(folder_base, "timestamps.txt"),
            os.path.join(folder_base, "pointcloud.pkl"),
        )
        match_list = match(video_stamps, point_data, num_nearest)[:400]
        num_matched = len(match_list)
        print(len(match_list), folder_base)
        
        # prepare to capture the video 
        video_cap = cv2.VideoCapture(os.path.join(folder_base, "video.mp4"))   # video_aug.mp4 or video.mp4
        video_fps = video_cap.get(5)
        cv2.namedWindow("fusion", 0)

        # initialize tracker
        # "(3,)<f4" means little-endian; an array with size (3, ); each element is a 4 bytes floats
        dtype_clusters = np.dtype({'names': ('num_points', 'center', 'size', 'avgV'),
                                'formats': ['<u4', '(3,)<f4', '(3,)<f4', '<f4']})    
        tracker = Tracker(dtype_clusters, radar_fps, max_age, min_hits)

        # iterate every frame
        while (1):
            cur_frame_idx = int(video_cap.get(1))       # index starts from 0
            cur_frame_idx = round(cur_frame_idx*20.0/video_fps)
            ret, frame = video_cap.read()
            previous_frame_name = " "
            if(ret):
                if cur_frame_idx < num_matched:
                    
                    # find nearby matched frames
                    matched_radar_idx =  match_list[cur_frame_idx]     # eg. [57 58 56]
                    idx_range = list(range(matched_radar_idx[0], matched_radar_idx[0] - overlay_num, -1))    

                    # 3D to 2D projection
                    x, y, z, v = np.array([]), np.array([]), np.array([]), np.array([])
                    for i in idx_range:
                        x = np.append(x, point_data[i]["Data"]["x"])
                        y = np.append(y, point_data[i]["Data"]["y"])
                        z = np.append(z, point_data[i]["Data"]["z"])
                        v = np.append(v, point_data[i]["Data"]["velocity"])
                    points_3d = np.array([x, y, z, v])         # 3d points in radar coordinate
                    uv, xyzV = from_3d_to_2d(points_3d, calib_param)   # 2d points in image coordinate

                    # FOV and velocity filter
                    filtering = [0<=i[0]<640 and 0<=i[1]<480 and j[2]<max_depth and abs(j[3])>= min_velocity for i, j in zip(uv, xyzV)]
                    uv, xyzV = uv[filtering], xyzV[filtering]
                    return_point_cloud = np.concatenate((uv, xyzV[..., 2:]), -1)

                    # clustering and tracking
                    clusters, labels = radar_dbscan(xyzV, dtype_clusters, dbscan_weights, dbscan_eps)
                    clusters = clusters[clusters["num_points"] >= num_pts_filter]
                    tracked_clusters = tracker.update(clusters)

                    # generate box proposals
                    xyxys = []
                    for cluster in tracked_clusters:   
                        center = cluster['center']  
                        size = cluster['size']
                        if max(size) < max_size:
                            # 3d->xywh
                            corners_3d = np.tile(center,(2,1)) + np.tile(size,(2,1)) * np.array([[1,1,0], [-1,-1,0]])/2    # shape (2, 3)
                            u, v = projection_xyr_to_uv(corners_3d.transpose(), calib_param)
                            x,y,w,h = (u[0]+u[1])/2, (v[0]+v[1])/2, u[0]-u[1], v[0]-v[1]

                            # box proposal position compensation
                            move = [h/5, w/2]
                            translations = [[0, move[0]*0.8]]
                            scales = [[1.2, 1.4]]
                            augments = np.array([ [[*i, 0, 0], [1, 1, *j]] for i in translations for j in scales])
                            xywh = np.tile(np.array([x,y,w,h]), (len(augments),1)) * augments[:, 1, :] + augments[:, 0, :]
                            xyxys.extend(xywh2xyxy(xywh))

                            # draw mapped 2d bboxes/point cloud
                            if show: 
                                for i in xyxys:
                                    cv2.rectangle(frame, (round(i[0]), round(i[1])), (round(i[2]), round(i[3])), (0,0,255))
                                for i in return_point_cloud:
                                    color = (0, int((1-i[2]/max_depth)*255), int(i[2]/max_depth*255))
                                    cv2.circle(frame, (round(i[0]), round(i[1])), 7, color, -1)

                    xyxys = np.array(xyxys)

                    if (cur_frame_idx % fps_divider == 0) and not show:
                        # save img, pointcloud, radar box proposal for each frame
                        dataset_folder = "./dataset"
                        frame_name = folder_base_list[folder_idx]+"-"+str(cur_frame_idx)

                        img_file = os.path.join(dataset_folder, "image", frame_name+".jpg")
                        cv2.imwrite(img_file, frame)

                        radarpoint_file = os.path.join(dataset_folder, "radar_point", frame_name+".pkl")
                        f = open(radarpoint_file, 'wb')
                        pickle.dump(return_point_cloud, f)

                        radarbox_file = os.path.join(dataset_folder, "radar_box", frame_name+".pkl")
                        f = open(radarbox_file, 'wb')
                        pickle.dump(xyxys, f)

                        f = open(os.path.join("./dataset/dataset.txt"), 'a+')
                        f.write(frame_name + "\n")
                        print(frame_name)

                cv2.imshow("fusion", frame)
                if cv2.waitKey(10) & 0xFF == ord("p"):
                    cv2.waitKey(10000000)
            else: 
                break

        video_cap.release()
        cv2.destroyAllWindows()
