from __future__ import division

from yolov3.models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from my_models import Network, define_yolo, init_yolo

import os
import time
import datetime
import argparse
import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.optim as optim

# torch.backends.cudnn.benchmark=True

def mode_selection(mode, img, paths): 
    # mode: [millieye, yolo, radar, auto]
    if mode in [0, 1, 2]:
        return mode
    if mode == 3:     # auto
        if img.mean()<0.1:
            return 0
        else:
            return 1


def evaluate(model, mode, model_mode, illumination, iou_thresh, nms_thresh, img_size, batch_size, test_list):
    """
    params
    ---
        - model, mode, iou_thresh, conf_thresh, nms_thres, img_size, batch_size

    return
    ---
        - precision, recall, AP, f1, ap_class, box_stat, pr_curve
    """
    model.eval()

    # Get dataloader
    dataset = MyDataset(mode=mode, illumination=illumination, augment=False, multiscale=False, test_list=test_list, dataset_folder="../data/our_dataset")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = [] 
    sample_metrics = []  # List of tuples (TP, confs, pred)
    box_stat = dict(before = [1], after = [1])

    for (paths, imgs, targets, radar_boxes, radar_maps) in tqdm.tqdm(dataloader, desc="Detecting objects"):
        
        with torch.no_grad():
            imgs = imgs.to(device)  # to gpu if any
            radar_maps = radar_maps.to(device)  
            radar_boxes = radar_boxes.to(device)  
            # _, outputs, _, _ = model(imgs, radar_maps, radar_boxes, targets.clone())

            model_mode_tmp = mode_selection(model_mode, imgs, paths)

            import time

            t = time.time()
            outputs = model(imgs, radar_maps, radar_boxes, model_mode_tmp)
            t = time.time()-t

            # output: tensor[N,8] -> outputs_reshape: List[tensor[n,7]]
            outputs_reshape = [None for _ in range(len(imgs))]  # batch_size
            for item in outputs.to(torch.device("cpu")):
                i = item[0].int()
                if outputs_reshape[i] is None:
                    outputs_reshape[i] = item[1:].unsqueeze(0)
                else:
                    outputs_reshape[i] = torch.cat((outputs_reshape[i], item[1:].unsqueeze(0)),0)

            # record boxes amount:
            for image_pred in outputs_reshape:
                box_stat["after"].append(len(image_pred) if not image_pred==None else 0)
            
        # Extract classes
        labels += targets[:, 1].tolist()

        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        sample_metrics += get_batch_statistics(
            outputs_reshape, targets, iou_threshold=iou_thresh
        )

    # Concatenate sample statistics
    if sample_metrics == []:
        true_positives, pred_scores, pred_labels, labels = np.array([0]), np.array([1]), np.array([1]), np.array([1])
    else:
        true_positives, pred_scores, pred_labels = [
            np.concatenate(x, 0) for x in list(zip(*sample_metrics))
        ]
    precision, recall, AP, f1, ap_class, pr_curve = ap_per_class(
        true_positives, pred_scores, pred_labels, labels
    )

    return precision, recall, AP, f1, ap_class, box_stat, pr_curve



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", type=int, 
        default=1, help="size of each image batch"
        )
    parser.add_argument(
        "--img_size", type=int, 
        default=416, help="size of each image dimension"
        )  
    parser.add_argument(
        "--yolo_cfg", type=str,
        default="config/yolov3-tiny-12.cfg"
        )
    parser.add_argument(
        "--classes_path", type=str,
        default="config/exdark.names", help="path to data config file"
        )
    parser.add_argument(
        "--checkpoint", type=str,
        default="./checkpoints/2_ckpt_best.pth"
        )   
    parser.add_argument(
        "--conf_thresh", type=float, 
        default=0.2, help="object confidence threshold"
        )
    parser.add_argument(
        "--iou_thresh", type=float, 
        default=0.5, help="mask threshold for outputs of the fusion network"
        )  
    parser.add_argument(    # check datasets.py whether comment lines about 'D'
        "--scene_mode", type=int,
        default=2, help="train and test mode: ['H'], ['L'], ['H', 'L'], ['D']"
        )       
    parser.add_argument(
        "--test_list", type=int,
        default=0, help="test scenem: 0, 1, 2, 3, 4"
        )
    parser.add_argument(
        "--model_mode", type=int,
        default=2, help="four mode: [millieye, yolo, radar, auto]"
        )

    opt = parser.parse_args()
    opt.scene_mode = [['H'], ['L'], ['H', 'L'], ['D']][opt.scene_mode]
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initiate model
    base_detector = define_yolo(opt.yolo_cfg)
    model = Network(base_detector, opt.conf_thresh).to(device)

    # load (yolo+module2+module3) from checkpoint
    model.load_state_dict(torch.load(opt.checkpoint))

    # parameter count
    total_num = sum(p.numel() for p in model.refinement_head.parameters())
    trainable_num = sum(p.numel() for p in model.refinement_head.parameters() if p.requires_grad)
    print(f"Total: {total_num}", f"Trainable: {trainable_num}")

    # compute mAP
    print("Compute mAP...")
    precision, recall, ap, f1, ap_class, box_stat, pr_curve = evaluate(
        model,
        mode = 'test',
        model_mode = opt.model_mode, 
        illumination = opt.scene_mode,
        iou_thresh = opt.iou_thresh,
        nms_thresh = 0.5,
        img_size = opt.img_size,
        batch_size = opt.batch_size,
        test_list = opt.test_list
    )

    # draw figures
    plt.figure(figsize=(6, 3))
    plt.subplot(111)
    p, r = np.array(pr_curve[0]), np.array(pr_curve[1])
    plt.plot(r, p)
    plt.title('pr-curve')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.tight_layout()
    
    os.makedirs("plot", exist_ok=True)
    plt.savefig(f"plot/pr_{opt.scene_mode}_{opt.conf_thresh}_{opt.iou_thresh}.jpg")
    plt.close()

    # print info        
    print(f"mAP: {ap.mean()}")
