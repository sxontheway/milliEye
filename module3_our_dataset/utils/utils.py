from __future__ import division
import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.ops import boxes as box_ops

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight.data)
    # print(m.__class__)


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y    


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.  
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics. 

    # Arguments 
        tp: True positives (list), e.g. array([0., 1., 1.]), 3 predicted bboxs
        conf: Objectness value from 0-1 (list), e.g. tensor([0.8247, 0.3907, 0.6466])  
        pred_cls: Predicted object classes (list), e.g. tensor([1., 0., 18.])   
        target_cls: True object classes (list), e.g. tensor([1., 2., 18.])  
    # Returns  
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = (pred_cls == c)     # only keep class c for predicted bboxs
        n_p = i.sum()  # Number of predicted objects
        n_gt = (target_cls == c).sum()  # Number of ground truth objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)


    # pr-curve for all classes added together
    filtering = []
    for i in range(len(tp)):
        if pred_cls[i] in unique_classes:
            filtering.append(True)
        else:
            filtering.append(False)
    tp, pred_cls = tp[filtering], pred_cls[filtering]

    n_p = len(tp) 
    n_gt = len(target_cls)

    if n_p == 0 or n_gt == 0:
        precision_curve, recall_curve = 0, 0
    else:
        # Accumulate FPs and TPs
        fpc = (1 - tp).cumsum()
        tpc = (tp).cumsum()

        # Recall and Precision
        recall_curve = tpc / (n_gt + 1e-16)
        precision_curve = tpc / (tpc + fpc)

    return p, r, ap, f1, unique_classes.astype("int32"), (precision_curve,recall_curve)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    """ 
    Compute true positives, predicted scores and predicted labels per sample 
    
    Argus:
    ---
    -outputs: list[tensor[n, 7]], len(outputs) = batch_size
    [x1, y1, x2, y2, obj_conf, obj_score, class_pred]  
    (x1, y1, x2, y2) is scaled to img_size 

    -targets: tensor with size (m, 6)   
    [image_i, class, x_center, y_center, w, h]    
    (x_center, y_center, w, h) is scaled to img_size
    """
    batch_metrics = []
    
    # iter for each image, image_i is the idx of the image in a batch
    for image_i in range(len(outputs)):

        if outputs[image_i] is None:
            continue

        output = outputs[image_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == image_i][:, 1:]     # filter for image_i
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            # iter for every detected box
            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If all targets are found, break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue
                
                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:# and pred_label==target_labels[box_index]:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of box1 (shape 1*4) with several box2 (shape n*4)
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)     # broadcast here
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, conf_thresh=0.01, nms_thresh=0.5):
    """
    CPU single-process version of NMS
    Removes detections with lower object confidence score than 'conf_thresh' and   
    performs Non-Maximum Suppression to further filter detections. 

    Parameters
    ---
    prediction: take yolov3 tiny for example, it is a tensor(batch_size, 2535, 85)

    Returns
    ---
    list[tensor(n, 7)], where n is the number of bboxes after NMS in an image
    7 for (x1, y1, x2, y2, object_conf, class_score, class_pred)
    Here (x1, y1, x2, y2) is scale to image size
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]

    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thresh]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue

        # Object confidence times class confidence
        score = image_pred[:, 4]    # *image_pred[:, 5:].max(1)[0] is optional: could drop mAP ~0.1
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            # detections[0, :4] is the bbox with highest score
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thresh
            label_match = detections[0, -1] == detections[:, -1]
            invalid = large_overlap & label_match

            # Merge overlapping bboxes by order of confidence: could increase mAP ~0.4
            # weights = detections[invalid, 4:5]
            # detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            
            # filter boxes
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


def non_max_suppression_cpp(prediction, conf_thresh, nms_thresh=0.5, detections_per_img=200):
    """
    This function can run on both CPU and GPU, depending on the prediction.device  
    Removes detections with lower object confidence score than 'conf_thresh' and performs
    Non-Maximum Suppression to further filter detections. 

    Parameters
    ---
    prediction: take yolov3-tiny (COCO) for example, it is a tensor(batch_size, 2535, 85)

    Returns
    ---
    list[tensor(n, 7+c)], where n is the number of bboxes after NMS in an image
    7+c for (x1, y1, x2, y2, object_conf, class_score, class_pred, scores_of_c_classes)
    Here (x1, y1, x2, y2) is scale to image size
    """
    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]

    for image_i, image_pred in enumerate(prediction):
        # Filter out low confidence boxes
        image_pred = image_pred[image_pred[:, 4] >= conf_thresh]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        
        # obtain class of each bbox
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float(), image_pred[:, 5:]), 1)

        boxes = detections[:, :4]   # tensor[n,4]
        scores = detections[:, 4]   # tensor[n], detections[:, 4]*class_confs.squeeze() is optional: could drop mAP ~0.1
        labels = detections[:, 6]   # tensor[n]

        keep = box_ops.batched_nms(boxes, scores, labels, nms_thresh)
        keep = keep[:detections_per_img]

        if len(keep) > 0:
            output[image_i] = detections[keep]

    return output


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    """ 
    pred_boxes: shape (batch_size, num_anchors, num_grids, num_grids, 4), e.g. (32,3,13,13,4)  
    pred_cls: shape (batch_size, num_anchors, num_grids, num_grids, num_classes), e.g.(32,3,13,13,12)  
    target: shape (num_bboxes_in_a_batch, 6)  
    anchors: e.g. tensor([[ 2.5312, 2.5625], [4.2188, 5.2812], [10.7500, 9.9688]], device='cuda:0')
    """
    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0) # batch size
    nA = pred_boxes.size(1) # num of anchors in a grid, 3 in yolo-tiny
    nC = pred_cls.size(-1)  # number of classes
    nG = pred_boxes.size(2) # number of grids

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou, here it only chooses the shape of anchor
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
