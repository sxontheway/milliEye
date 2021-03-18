from __future__ import division

from yolov3.models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from my_models import Network, define_yolo, init_yolo

import os
import sys
import time
import datetime
import argparse
import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def evaluate(model, list_path, iou_thresh, conf_thresh, nms_thresh, img_size, batch_size):
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
    dataset = ListDataset(list_path, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=32,
        collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    labels = [] 
    sample_metrics = []  # List of tuples (TP, confs, pred)
    box_stat = dict(before = [1], after = [1])

    for batch_i, (_, imgs, targets) in enumerate(
        tqdm.tqdm(dataloader, desc="Detecting objects")):

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)   # outputs is on cpu

            # output: tensor[N,8] -> outputs_reshape: List[tensor[n,7]]
            outputs_reshape = [None for _ in range(len(imgs))]
            for item in outputs.to(torch.device("cpu")):
                i = item[0].int()
                if outputs_reshape[i] is None:
                    outputs_reshape[i] = item[1:].unsqueeze(0)
                else:
                    outputs_reshape[i] = torch.cat((outputs_reshape[i], item[1:].unsqueeze(0)),0)

            # record boxes amount:
            for image_pred in outputs_reshape:
                box_stat["after"].append(len(image_pred) if not image_pred==None else 0)
        
        # Extract class labels
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
        default=32, help="size of each image batch"
    )
    parser.add_argument(
        "--data_config", type=str,
        default="config/exdark.data", help="path to data config file",  # choose to test on ExDark/coco/mixed
    )
    parser.add_argument(
        "--classes_path", type=str,
        default="config/exdark.names", help="path to file of class name"
    )
    parser.add_argument(
        "--conf_thresh", type=float, 
        default=0.01, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou_thresh", type=float,
        default=0.5, help="iou threshold required to qualify as detected",
    )
    parser.add_argument(
        "--yolo_cfg", type=str,
        default="config/yolov3-tiny-12.cfg", help="train and test mode"
    )  
    parser.add_argument(
        "--img_size", type=int, 
        default=416, help="size of each image dimension"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="./checkpoints/module2_best_mixed.pth", help="interval between saving model weights"
    )   # ckpt 71 or 87, 
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get classes
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(opt.classes_path)

    # Initiate model
    base_detector = define_yolo(opt.yolo_cfg)
    model = Network(base_detector, opt.conf_thresh).to(device)

    # load (yolo+module2) from checkpoint
    model.load_state_dict(torch.load(opt.checkpoint))

    print("Compute mAP...")

    precision, recall, ap, f1, ap_class, box_stat, pr_curve = evaluate(
        model,
        list_path = valid_path,
        iou_thresh = opt.iou_thresh,
        conf_thresh = 0.01, # not used
        nms_thresh = 0.5,   # not used
        img_size = opt.img_size,
        batch_size = opt.batch_size,
    )
    # draw figures
    print(f"img_number: {len(box_stat['after'])}, sample_number: {len(pr_curve[0])}")
    before, after = np.array(box_stat["before"]), np.array(box_stat["after"])
    p, r, conf = np.array(pr_curve[0]), np.array(pr_curve[1]), np.array(pr_curve[2])
    
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # mpl.use('Agg')
    mpl.rcParams['font.size'] = 20
    mpl.rcParams['figure.titlesize'] = 'medium'
    plt.figure(figsize=(10,5))

    plt.subplot(111)
    plt.plot(r,p,lw=3)
    plt.title('P-R Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.tight_layout()
    os.makedirs("plot", exist_ok=True)
    dataset_name = opt.checkpoint.split("/")[2]
    plt.savefig(f"plot/module2/{opt.iou_thresh}_{opt.conf_thresh}.jpg")
    plt.close()

    # print info
    for i, c in enumerate(ap_class):    # ap_class are the number in the label that indicating every class
        print(f"+ Class {c} ({class_names[i]})".ljust(30) + f"-AP: {ap[i]:.3f} -Precision:{precision[i]:.3f} -Recall:{recall[i]:.3f}")
    print(f"mAP: {ap.mean()}")
