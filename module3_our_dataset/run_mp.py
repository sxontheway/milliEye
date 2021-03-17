######################################################
# A script for real-time demo using cpu/gpu pipeline #
######################################################

from __future__ import division

from yolov3.models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from my_models import Network, define_yolo, init_yolo
from data_collection.utils.utils import *

import os
import time, datetime
import numpy as np
import cv2
import re
import pickle
import argparse
import multiprocessing as mp

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


class Args():   # an empty to store params
    def __init__(self):
        pass


def pre_process(q: mp.Queue(), event, folder_base, point_data, match_list, tracker, args):
    """
    Arguments:
        - video_cap, point_data, match_list, args
    Return:
        - img, radar_map, radar_box
    """

    video_cap = cv2.VideoCapture(os.path.join(folder_base, "video.mp4"))   # video_aug.mp4 or video.mp4
    video_fps = video_cap.get(5)

    try:
        while 1:
            
            t = time.time()
            ret, frame = video_cap.read()

            if ret: 
                cur_frame_idx = int(video_cap.get(1))       # index starts from 0
                cur_frame_idx = round(cur_frame_idx*20.0/video_fps)

                if cur_frame_idx < len(match_list):

                    ################## Radar Tracking ###############
                    #################################################

                    # find nearby matched frames
                    matched_radar_idx = match_list[cur_frame_idx]     # eg. [57 58 56]
                    idx_range = list(range(matched_radar_idx[0], matched_radar_idx[0] - args.overlay_num, -1))    

                    # 3D to 2D projection
                    x, y, z, v = np.array([]), np.array([]), np.array([]), np.array([])
                    for i in idx_range:
                        x = np.append(x, point_data[i]["Data"]["x"])
                        y = np.append(y, point_data[i]["Data"]["y"])
                        z = np.append(z, point_data[i]["Data"]["z"])
                        v = np.append(v, point_data[i]["Data"]["velocity"])
                    points_3d = np.array([x, y, z, v])         # 3d points in radar coordinate
                    uv, xyzV = from_3d_to_2d(points_3d, args.calib_param)   # 2d points in image coordinate

                    # FOV and velocity filter
                    filtering = [0<=i[0]<640 and 0<=i[1]<480 and j[2]<args.max_depth and abs(j[3])>= args.min_velocity for i, j in zip(uv, xyzV)]
                    uv, xyzV = uv[filtering], xyzV[filtering]
                    return_point_cloud = np.concatenate((uv, xyzV[..., 2:]), -1)

                    # clustering and tracking
                    clusters, labels = radar_dbscan(xyzV, args.dtype_clusters, args.dbscan_weights, args.dbscan_eps)
                    clusters = clusters[clusters["num_points"] >= args.num_pts_filter]
                    tracked_clusters = tracker.update(clusters)

                    # generate box proposals
                    xyxys = []
                    for cluster in tracked_clusters:   
                        center = cluster['center']  
                        size = cluster['size']
                        if max(size) < args.max_size:
                            # 3d->xywh
                            corners_3d = np.tile(center,(2,1)) + np.tile(size,(2,1)) * np.array([[1,1,0], [-1,-1,0]])/2    # shape (2, 3)
                            u, v = projection_xyr_to_uv(corners_3d.transpose(), args.calib_param)
                            x,y,w,h = (u[0]+u[1])/2, (v[0]+v[1])/2, u[0]-u[1], v[0]-v[1]

                            # box proposal position compensation
                            move = [h/5, w/2]
                            translations = [[0, move[0]*0.8]]
                            scales = [[1.2, 1.4]]
                            augments = np.array([ [[*i, 0, 0], [1, 1, *j]] for i in translations for j in scales])
                            xywh = np.tile(np.array([x,y,w,h]), (len(augments),1)) * augments[:, 1, :] + augments[:, 0, :]
                            xyxys.extend(xywh2xyxy(xywh))

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
                        if len(boxes)>0:    # if there still are boxes after filtering
                            radar_box = torch.zeros(len(boxes), 5)
                            radar_box[:, 0] = 0  # add a place holder 0 for index in a batch
                            radar_box[:, 1:] = boxes 

                    # radar point
                    radar_map = transforms.ToTensor()(plot_radar_heatmap(return_point_cloud.transpose(), (w, h))).float()
                    radar_map, pad = pad_to_square(radar_map, 0)   

                    # write into pipe
                    img = torch.stack([resize(x, 416) for x in img.unsqueeze(0)])
                    radar_map = radar_map.unsqueeze(0)
                    model_mode = mode_selection(args.model_mode, img)
                    return_list = [img, frame, radar_map, radar_box, cur_frame_idx, model_mode]
                    
                    q.put(return_list)

                    if cur_frame_idx == 1:  # inference of the first frame on gpu is very slow
                        event.wait()
                    if q.qsize() > 2:   # avoid the process being blocked if the queue is full
                        q.get()
                        
                    # print(f"pre_process_fps = {1/float(time.time()-t)}")

                else:
                    video_cap.release()

    except:
        video_cap.release()



def inference_post_process(q: mp.Queue(), event, model, args):
    """
    inference and result plotting
    """
    try:

        model = model.to(args.device)

        while 1:    
            print("*********", q.qsize())
            if q.qsize() > 0:

                t = time.time()
                img, frame, radar_map, radar_box = q.get()
                img = img.to(args.device)
                radar_map = radar_map.to(args.device)  
                radar_box = radar_box.to(args.device)  

                with torch.no_grad(): 
                    outputs = model(img, radar_map, radar_box)[:, 1:].cpu()  # tensor[N,7]
                    event.set()
                gpu_fps = 1/float(time.time()-t)
                print(f"gpu_fps = {gpu_fps}")

                # post=processing
                if outputs[0] is not None:
                    rescale_boxes(outputs, 416, frame.shape[:2])    # Rescale boxes to original image

                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs:
                        if int(cls_pred) in [0]: 
                            cv2.rectangle(frame, (x1.round(), y1.round()), (x2.round(), y2.round()), (0, 255, 255), 2)
                    
                cv2.putText(frame, f"fps:{gpu_fps:.1f}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.imshow("fusion", frame)
                if cv2.waitKey(1) & 0xFF == ord("p"):
                    cv2.waitKey(10000000)
    except:
        cv2.destroyAllWindows()


def mode_selection(mode, img): 
    # mode: [millieye, yolo, radar, auto]
    if mode in [0, 1, 2]:
        return mode
    if mode == 3:     # auto
        if img.mean() < 0.08:
            return 0
        else:
            return 1


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
        default=0.25, help="object confidence threshold"
        )
    parser.add_argument(
        "--model_mode", type=int,
        default=3, help="four mode: [millieye, yolo, radar, auto]"
        )
    opt = parser.parse_args()
    print(opt)

    # params、
    args = Args()
    args.radar_fps = 20                  # the fps of radar data                         
    args.num_nearest = 3                 # for each video frame, find "args.num_nearest" radar frames 
    args.overlay_num = 2                 # the point cloud from "args.overlay_num" radar frames are aggregared together
    # ---   
    args.dbscan_weights = [2,1,3,1]      # weights of xyzV
    args.dbscan_eps = 1.5                # the hyper-param in DBSCAN
    # ---   
    args.num_pts_filter = 5              # remove clusters whose points numbers are below the threshold 
    args.min_velocity = 0.1              # the velocity threshold of points
    args.max_size = 20                   # the upper bound of the proposed 3d box sizes
    args.max_depth = 10                  # the max depth of points 
    # ---   
    args.max_age = 4                       # keep tracking for how many frames                 
    args.min_hits=4                      # start tracking for how many frames
    args.calib_param = load_calib("./data_collection/yaml/calib_FOV90.yaml")     # calibration matrix
    # ---   
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.model_mode = opt.model_mode

    # initialize model
    base_detector = define_yolo(opt.yolo_cfg)
    model = Network(base_detector, opt.conf_thresh)
    model.load_state_dict(torch.load(opt.checkpoint))   # load (yolo+module2+module3) from checkpoint
    model.refine_threshold_radar = 0.56
    model.eval()
    model = model.to(args.device)
    torch.backends.cudnn.benchmark = True

    # iterate every folder
    raw_data_folder = "./data_collection/data/"    # "../data/data/"
    folder_base_list = sorted([string for string in os.listdir(raw_data_folder) if re.match(r'(.*)-(.*)', string)])
    folder_idxs = range(len(folder_base_list))[0:]

    cv2.namedWindow("fusion", 0)

    for folder_idx in folder_idxs:
        folder_base = raw_data_folder + folder_base_list[folder_idx]

        # do synchronization by matching timestamps
        video_stamps, point_data = load_data(           # video_stamps starts from 0
            os.path.join(folder_base, "timestamps.txt"),
            os.path.join(folder_base, "pointcloud.pkl"),
        )
        match_list = match(video_stamps, point_data, args.num_nearest)[:400]

        # initialize tracker
        # "(3,)<f4" means little-endian; an array with size (3, ); each element is a 4 bytes floats
        args.dtype_clusters = np.dtype({'names': ('num_points', 'center', 'size', 'avgV'),
                                'formats': ['<u4', '(3,)<f4', '(3,)<f4', '<f4']})    
        tracker = Tracker(args.dtype_clusters, args.radar_fps, args.max_age, args.min_hits)

        mp.set_start_method(method='spawn', force=True)
        queue = mp.Queue(maxsize=3)
        event = mp.Event()

        # use another process for pre-processing
        processes = []        
        processes.append(mp.Process(target=pre_process, args=(queue, event, folder_base, point_data, match_list, tracker, args)))
        # processes.append(mp.Process(target=inference_post_process, args=(queue, event, model, args)))

        for process in processes:
            process.daemon = True
            process.start()

        while 1:    

            # wait until the pre-processing is ready
            if queue.qsize() > 0:   

                t = time.time()
                img, frame, radar_map, radar_box, cur_frame_idx, model_mode = queue.get()
                img = img.to(args.device)
                radar_map = radar_map.to(args.device)  
                radar_box = radar_box.to(args.device)  

                with torch.no_grad(): 
                    outputs = model(img, radar_map, radar_box, model_mode)[:, 1:].cpu()  # tensor[N,7]
                event.set()
                gpu_fps = 1/float(time.time()-t)
                print(f"gpu_fps = {gpu_fps}")

                # post=processing
                keep = box_ops.batched_nms(outputs[:, :4], outputs[:, 4], outputs[:, 6], 0.3)
                outputs = outputs[keep]
                if len(outputs) > 0:
                    rescale_boxes(outputs, 416, frame.shape[:2])    # Rescale boxes to original image
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs:
                        if int(cls_pred) in [0]: 
                            cv2.rectangle(frame, (x1.round(), y1.round()), (x2.round(), y2.round()), (0, 255, 255), 2)
                            cv2.putText(frame, f"{conf:.2f}", (x1.round(), y1.round()-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                cv2.putText(frame, f"fps:{gpu_fps:.1f}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.imshow("fusion", frame)
                if cv2.waitKey(1) & 0xFF == ord("p"):
                    cv2.waitKey(10000000)
                if cur_frame_idx >= len(match_list)-1:
                    break

        for process in processes:
            process.terminate()  # terminate child process manually
            event.clear()

    cv2.destroyAllWindows()

