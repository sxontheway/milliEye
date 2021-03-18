from __future__ import division

from yolov3.models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def evaluate(model, mode, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    """
    params
    ---
        - model, mode, iou_thres, conf_thres, nms_thres, img_size, batch_size

    return
    ---
        - precision, recall, AP, f1, ap_class, box_stat, pr_curve
    """
    model.eval()

    # Get dataloader
    dataset = ExDarkDataset(mode, coco_detector = False, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=32,
        collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    labels = [] 
    sample_metrics = []  # List of tuples (TP, confs, pred)
    box_stat = dict(before = [], after = [])

    for batch_i, (_, imgs, targets) in enumerate(
        tqdm.tqdm(dataloader, desc="Detecting objects")
    ):

        # Extract class labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            _, outputs = model(imgs)
            outputs = outputs.to(torch.device("cpu"))

            # boxes amount before NMS
            for image_pred in outputs:
                box_stat["before"].append(len(image_pred[image_pred[:, 4] >= conf_thres]))
            
            # NMS
            outputs = non_max_suppression_cpp(
                outputs, conf_thresh=conf_thres, nms_thresh=nms_thres
            )
            for i, output in enumerate(outputs):
                if output is not None:
                    outputs[i] = output[:, :7]

            # boxes amount after NMS
            for image_pred in outputs:
                box_stat["after"].append(len(image_pred) if not image_pred==None else 0)

        sample_metrics += get_batch_statistics(
            outputs, targets, iou_threshold=iou_thres
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




'''
Yolo trained on COCO, test on Exdark
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", type=int, 
        default=32, help="size of each image batch"
    )
    parser.add_argument(
        "--model_def", type=str,
        default="config/yolov3-tiny-12.cfg", help="path to model definition file",
    )
    parser.add_argument(
        "--weights_path", type=str,
        default="weights/best_mixed.pt", help="path to weights file",
    )
    parser.add_argument(
        "--classes_path", type=str,
        default="config/exdark.names", help="path to class label file",
    )
    parser.add_argument(
        "--iou_thres", type=float,
        default=0.5, help="iou threshold required to qualify as detected",
    )
    parser.add_argument(
        "--conf_thres", type=float, 
        default=0.01, help="object confidence threshold"
    )
    parser.add_argument(
        "--nms_thres", type=float,
        default=0.5, help="iou thresshold for non-maximum suppression",
    )
    parser.add_argument(
        "--img_size", type=int, 
        default=416, help="size of each image dimension"
    )
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    elif opt.weights_path.endswith(".pt"):
        # load ultralytics yolov3 weights
        param = torch.load(opt.weights_path)["model"]   # param: a collections.OrderedDict
        module_list = model.state_dict()    # module_list: a collections.OrderedDict
        para_name = list(param)
        for i, name in enumerate(module_list):
            module_list[name] = param[para_name[i]]
        model.load_state_dict(module_list)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")

    precision, recall, ap, f1, ap_class, box_stat, pr_curve = evaluate(
        model,
        mode = "test",
        iou_thres = opt.iou_thres,
        conf_thres = opt.conf_thres,
        nms_thres = opt.nms_thres,
        img_size = opt.img_size,
        batch_size = opt.batch_size,
    )

    # draw figures
    print(f"img_number: {len(box_stat['after'])}, sample_number: {len(pr_curve[0])}")
    before, after = np.array(box_stat["before"]), np.array(box_stat["after"])
    p, r, conf = np.array(pr_curve)

    import matplotlib as mpl
    import matplotlib.pyplot as plt
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
    plt.savefig(f"plot/yolo_mixed/{opt.iou_thres}_{opt.conf_thres}.jpg")
    plt.close()


    # print info
    ExDark_mAP = 0
    choosen_class = [*range(12)]  # indices of chosen ExDark classes in coco.names 
    class_names = load_classes(opt.classes_path)  # coco class names  
    for i, c in enumerate(ap_class):
        if c in choosen_class:
            print(f"+ Class {c} ({class_names[c]})".ljust(30) + f"-AP: {ap[i]:.3f} -Precision:{precision[i]:.3f} -Recall:{recall[i]:.3f}")
            ExDark_mAP += ap[i]
        else:
            print(f"- Class {c} ({class_names[c]})".ljust(30) + f"-AP: {ap[i]:.3f} -Precision:{precision[i]:.3f} -Recall:{recall[i]:.3f}")
    print(f"mAP_chosen classes:{ExDark_mAP/len(choosen_class)}")
    print(f"mAP_all classes: {ap.mean()}")  # mAP on classes included in annotations. Go to datasets.py to modify.
