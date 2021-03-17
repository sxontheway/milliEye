####################################################
# A script for real-time demo using single process #
####################################################

from __future__ import division

from yolov3.models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from my_models import Network, define_yolo, init_yolo

import os
import time, datetime
import numpy as np
import cv2
import re
import pickle
import argparse
from data_collection.utils.utils import *

import torch
from torch.utils.data import DataLoader
from torchvision import datasets


def xywh2xyxy(x):
    y = np.empty(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str,
        default="./checkpoints/2_ckpt_best.pth", help="interval between saving model weights"
        )  
    parser.add_argument(
        "--yolo_cfg", type=str,
        default="config/yolov3-tiny-12.cfg", help="train and test mode"
        )
    parser.add_argument(
        "--conf_thresh", type=float, 
        default=0.2, help="object confidence threshold"
        )
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    calib_param = load_calib("./data_collection/yaml/calib_FOV90.yaml")     # calibration matrix

    # Initiate model
    base_detector = define_yolo(opt.yolo_cfg)
    model = Network(base_detector, opt.conf_thresh)
    model.load_state_dict(torch.load(opt.checkpoint))   # load (yolo+module2+module3) from checkpoint
    model.refine_threshold_radar = 0.56
    model.eval()
    torch.backends.cudnn.benchmark = True

    # iterate every folder
    raw_data_folder = "./data_collection/data/"    # "../data/data/"
    folder_base_list = sorted([string for string in os.listdir(raw_data_folder) if re.match(r'(.*)-(.*)', string)])
    folder_idxs = range(len(folder_base_list))[5:]

    for folder_idx in folder_idxs:
        folder_base = raw_data_folder + folder_base_list[folder_idx]

        # do synchronization by matching timestamps
        video_stamps, point_data = load_data(           # video_stamps starts from 0
            os.path.join(folder_base, "timestamps.txt"),
            os.path.join(folder_base, "pointcloud.pkl"),
        )
        match_list = match(video_stamps, point_data, num_nearest)[:400]
        num_matched = len(match_list)
        
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

            if(ret):
                if cur_frame_idx < num_matched:

                    ################## Radar Tracking ###############
                    #################################################
                    # find nearby matched frames
                    t = time.time()
                    matched_radar_idx = match_list[cur_frame_idx]     # eg. [57 58 56]
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
                            # for i in xyxys:
                            #     cv2.rectangle(frame, (round(i[0]), round(i[1])), (round(i[2]), round(i[3])), (0,0,255), 2)
                            # for i in return_point_cloud:
                            #     color = (0, int((1-i[2]/max_depth)*255), int(i[2]/max_depth*255))
                            #     cv2.circle(frame, (round(i[0]), round(i[1])), 7, color, -1)
                            
                    tracking_fps = 1/float(time.time()-t)

                    ################## Preprocessing ###############
                    ################################################
                    # img
                    img = transforms.ToTensor()(frame)
                    _, h, w = img.shape     # (h, w) is the original size 
                    img, pad = pad_to_square(img, 0)    
                    _, padded_h, padded_w = img.shape   # img.shape is the padded size

                    # radar box
                    boxes = torch.tensor(xyxys)
                    radar_box = torch.zeros(len(boxes), 5)
                    if len(boxes) > 0:
                        # Adjust for added padding
                        boxes[:, 0] += pad[0]    # last dim
                        boxes[:, 2] += pad[1]    # pad[1] == pad[0]
                        boxes[:, 1] += pad[2]    # 2nd to last dim
                        boxes[:, 3] += pad[3]    # pad[3] == pad[2]
                        
                        # scale to (0, 1) 
                        boxes = torch.clamp(boxes/padded_h, 0, 1)  # padded_h == padded_w
                        boxes = boxes[torch.logical_and(boxes[:, 0]<boxes[:, 2], boxes[:, 1]<boxes[:, 3])]
                        if len(boxes)>0:
                            radar_box = torch.zeros(len(boxes), 5)
                            radar_box[:, 0] = 0  # add a place holder 0 for index in a batch
                            radar_box[:, 1:] = boxes 

                    # radar point
                    radar_map = transforms.ToTensor()(plot_radar_heatmap(return_point_cloud.transpose(), (w, h))).float()
                    radar_map, pad = pad_to_square(radar_map, 0)   

                    # move to gpu if any
                    img = torch.stack([resize(x, 416) for x in img.unsqueeze(0)]).to(device)
                    radar_map = radar_map.unsqueeze(0).to(device)  
                    radar_box = radar_box.to(device)  
                    preprocess_fps = 1/float(time.time()-t)

                    ################## CNN and show ###############
                    ###############################################
                    t = time.time()
                    with torch.no_grad(): 
                        model = model.to(device)
                        outputs = model(img, radar_map, radar_box)[:, 1:].cpu()  # tensor[N,7]
                        keep = box_ops.batched_nms(outputs[:, :4], outputs[:, 4], outputs[:, 6], 0.3)
                        outputs = outputs[keep]

                    gpu_time = float(time.time()-t)
                    gpu_fps = 1/gpu_time
                    t = time.time()

                    # Draw bounding boxes and labels of detections
                    if len(outputs) > 0:
                                        
                        # Rescale boxes to original image
                        rescale_boxes(outputs, 416, frame.shape[:2])
    
                        for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs:

                            if int(cls_pred) in [0]: 
                                cv2.rectangle(frame, (x1.round(), y1.round()), (x2.round(), y2.round()), (0, 255, 255), 2)
                    
                    cv2.putText(frame, f"fps:{gpu_fps:.1f}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    postprocess_fps = 1/float(time.time()-t)
                    print(tracking_fps, preprocess_fps, gpu_fps, postprocess_fps)
                    
                cv2.imshow("fusion", frame)
                if cv2.waitKey(10) & 0xFF == ord("p"):
                    cv2.waitKey(10000000)

            else: 
                break

        video_cap.release()
        cv2.destroyAllWindows()
