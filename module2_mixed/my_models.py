import numpy as np
import random
import torch
from torch import nn
from torchvision.ops import ps_roi_align

import os, time
from torch.multiprocessing import Pool, Manager
from yolov3.models import Darknet
from utils.utils import *


def define_yolo(model_def):
    """
    return
    ---
    a Darknet class object: yolo    
    the forward function of yolo returns:
        -(featuremap, yolo_outputs)         # for inference
        -(loss, featuremap, yolo_outputs)   # for training 
    """
    yolo = Darknet(model_def)

    return yolo


def init_yolo(model, weights_path):
    """
    load weight for yolo
    """
    if weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(weights_path)
    elif weights_path.endswith(".pt"):
        # load ultralytics yolov3 weights
        param = torch.load(weights_path)["model"]   # param: a collections.OrderedDict
        module_list = model.state_dict()    # module_list: a collections.OrderedDict
        para_name = list(param)
        for i, name in enumerate(module_list):
            module_list[name] = param[para_name[i]]
        model.load_state_dict(module_list)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(weights_path))


class fcn_layers(nn.Module):
    """
    CNN layers applied on feature maps to generate score maps, 
    followed by the RoI-wise cropping operation. 
    ---
    params for __init__():
        channels: e.g. (256, 490)
    forward():
        Input: feature maps
        Output: feature maps
    """
    def __init__(self, channels):
        super(fcn_layers, self).__init__()
        self.net = nn.Sequential()

        for i in range(len(channels)-1):
            in_channels, out_channels = channels[i], channels[i+1]
            self.net.add_module(
                f"conv_{i}",
                nn.Conv2d(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = (1,1),
                    stride = (1,1)
                    )
                )
            self.net.add_module(f"batch_norm_{i}", nn.BatchNorm2d(out_channels, momentum=0.1))
            self.net.add_module(f"leaky_{i}", nn.LeakyReLU(0.1))

    def forward(self, x):
        return self.net(x)


class self_attention(nn.Module):
    """
    Self-attention layer
    """
    def __init__(self, channels):
        super(self_attention, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(channels, channels),           
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        y = self.net(x)
        return x*y


class refinement_head(nn.Module):
    """
    part of the refinement head that is applied to single sensing modality
    ---
    params for __init__():
        -channels: e.g. (490, 256, c+1)
    forward():
        -inputs: feature maps, e.g. tensor(n*10*7*7)
    """
    def __init__(self, channels):
        super(refinement_head, self).__init__()
        self.net0 = nn.Sequential(
            nn.Linear(channels[0], channels[1]),    # 490->256
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5)
        )
        self.net1 = nn.Sequential(
            nn.Linear(channels[1], 4),    # 256->4
        )
        self.net2 = nn.Sequential(
            nn.Linear(channels[1], channels[2]),    # 256->(c+1)
            nn.Sigmoid()
        )

    def forward(self, img_maps):
        img_flatten = img_maps.flatten(start_dim=1)
        tmp = self.net0(img_flatten)
        box_location = self.net1(tmp)
        class_vector = self.net2(tmp)
        return box_location, class_vector
    

class ensemble_head(nn.Module):
    """
    Fuse two feature vectors using attention-based fusion
    ---
    params for __init__():
        -channels (list): number of neuros in each layer, e.g. (2, 16, 208, 2)
        -use_activation: if there is a softmax during the calculation of loss function,  
        then this should be set to False 
    ---
    forward():
        -inputs: two vectors. e.g., tensor(n, c+1), tensor(n, c+1)
        -outputs: mask vector, e.g. tensor(n*2)
    """

    def __init__(self, channels, use_activation=True):
        super(ensemble_head, self).__init__()
        self.use_activation = use_activation
        self.fc1 = nn.Sequential(
            nn.Linear(channels[0], channels[1]),
            nn.LeakyReLU(0.1)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(channels[2], channels[3]),
            nn.LeakyReLU(0.1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, refinement_vector, yolo_vector):
        x = torch.stack((refinement_vector, yolo_vector), -1)     # n*(c+1)*channels[0]
        x = self.fc1(x)     # n*(c+1)*channels[1]
        x = x.flatten(start_dim=1)      # n*channel[2]
        x = self.fc2(x)     # n*channel[3]
        if self.use_activation == True:
            x = self.softmax(x)

        return x

class FocalLoss(nn.Module):

    def __init__(self, device, alpha, gamma=2, reduction="sum"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.device = device

    def forward(self, inputs, labels):
        """
        inputs: tensor[n,2]
        labels: tensor[n,2], in one-hot coding form
        """
        alpha_pos = torch.full((labels.shape[0],1), self.alpha).to(self.device)        
        alpha_neg = torch.full((labels.shape[0],1), 1-self.alpha).to(self.device)
        alpha = torch.where(labels[:, 1:2]==1, alpha_pos, alpha_neg)
        
        probs = (inputs*labels).sum(1).view(-1,1)   # shape(n,1)
        log_p = probs.log()     # shape(n,1)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 

        if self.reduction == "mean":
            loss = batch_loss.mean()
        if self.reduction == 'sum':
            loss = batch_loss.sum()
        return loss


def obtain_iou_labels(boxes, targets, multi_boxes=True):
    """
    Return the max iou with ground truths for each bounding box 
    ---
    Argus
    ---
        -boxes, params[0]: tensor[p,6], 6 for [image_i, class, x1, y1, x2, y2]
        -targets, param[1]: tensor[q,6], 6 for [image_i, class, x1, y1, x2, y2]
        -multi_boxes: whether allows several positive samples for one GT box

    Return
    ---
        -iou_labels: tensor[k,1]
        -target_location: tensor[k, 4], 4 for [x1, y1, x2, y2]
    """
    image_index = boxes[:, :1]  
    pred_boxes = boxes[:, 2:]
    pred_classes = boxes[:, 1:2]        
    detected_boxes = []
    iou_labels = torch.zeros((len(image_index), 1))
    target_location = torch.zeros((len(image_index), 4))

    for box_i in range(len(boxes)):

        # choose annotations
        # filtering = [bool(a) for a  in zip(targets[:, 0] == image_index[box_i])]   # the same image
        filtering = [bool(a and b) for a,b in zip(
            targets[:, 0] == image_index[box_i],    # the same image
            targets[:, 1] == pred_classes[box_i]    # the same class 
        )]

        # If no target passes filter
        if (True not in filtering):
            continue
        # If all targets are found, break
        if multi_boxes is False and len(detected_boxes) == len(targets):
            break   
        
        pred_box = pred_boxes[box_i]    # torch.Size([4])
        target_boxes = targets[filtering][:, 2:]    # torch.Size([m, 4])
        ious = bbox_iou(pred_box.unsqueeze(0), target_boxes)
        if len(ious) > 0:
            iou, target_index = ious.max(0)
            if (target_index not in detected_boxes) or multi_boxes:
                iou_labels[box_i] = iou    
                target_location[box_i] = target_boxes[target_index]
                if iou > 0.7:
                    detected_boxes += [target_index]

    return iou_labels, target_location         


def box_regress(regress_param, roi_location):
    """
    -regress_param: n*4  
    -roi_location: n*4, xyxy, scaled to img size  
    -xyxy_regressed: n*4, xyxy, scaled to img size  
    """
    x, y, w, h = xyxy2xywh(roi_location).t()
    x_regressed = regress_param[:, 0]*w + x
    y_regressed = regress_param[:, 1]*h + y
    w_regressed = torch.exp(regress_param[:, 2]) * w
    h_regressed = torch.exp(regress_param[:, 3]) * h
    xyxy_regressed = xywh2xyxy(torch.stack((x_regressed, y_regressed, w_regressed, h_regressed), 1))

    return xyxy_regressed


def regression_loss(regress_param, target_location, roi_location):
    """
    -regress_param: n*4  
    -target_location: n*4, xyxy, scaled to img size  
    -roi_location: n*4, xyxy, scaled to img size  
    """
    x,y,w,h = xyxy2xywh(roi_location).t()
    xt,yt,wt,ht = xyxy2xywh(target_location).t()

    p01 = torch.stack( ((xt - x)/(w+1e-16), (yt - y)/(h+1e-16)), -1)
    p23 = torch.stack( (torch.log(wt/w + 1e-16), torch.log(ht/h + 1e-16)), -1)  
    loss_xy = torch.nn.SmoothL1Loss(reduction="sum")(p01, regress_param[:, :2])
    loss_wh = torch.nn.SmoothL1Loss(reduction="sum")(p23, regress_param[:, 2:])

    return loss_xy, loss_wh


class Network(nn.Module):

    def __init__(self, base_detector, conf_thresh):
        super(Network, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_thresh = conf_thresh
        self.seen = 0
        self.iou_thresh = (0.3, 0.7)
        self.alpha = 0.75   
        self.balance_fac = 5
        self.loss_lambda = (15, 5)
        self.refine_threshold = 0
        self.class_num = 12

        self.base_detector = base_detector.eval()
        self.fcn_layers = fcn_layers((256, 490))
        self.refinement_head = refinement_head((490, 256, self.class_num+1)) 
        self.ensemble_head = ensemble_head((2, 32, 32*(self.class_num+1), 2))

    def forward(self, images, targets=None):
        """
        Arguments:
            -images: Tensor[N,C,H,W], here H=W=416
            -targets: Tensor[m, 6], m is the number of annotated bboxes in a batch.  
            6 for [index_in_a_batch, class_num, x_center, y_center, w, h]
            Here (x_center, y_center, w, h) is scale to (0, 1)

        Returns:
            -loss: float
            -output: the new score for every bbox, Tensor[m, 8].
                8 for [image_i, x1, y1, x2, y2, object_conf, class_score, class_pred]
                Here (x1, y1, x2, y2) is scale to image size (e.g., 416)
        """
        ########################################
        # Get candidate boxes from base detector
        ########################################
        # for tiny yolov3, feature_map: (1,256,26,26); output_tensor: (1, 2535, 85). No gradient.
        feature_map, output_tensor = self.base_detector(images)   

        # NMS in a batch manner, CPP version
        detections = non_max_suppression_cpp(output_tensor.cpu(), conf_thresh = self.conf_thresh)

        # detections: List[tensor[n,7+c]] -> boxes: tensor[N,8+c], (8+c) for 
        # (image_i, x1, y1, x2, y2, object_conf, class_score, class_pred, scores_of_c_classes), xyxy are scaled to img size
        boxes = []
        for image_i, detection_i in enumerate(detections):  # iter through a batch
            if detection_i is not None:
                boxes_i = torch.zeros((len(detection_i), 8+self.class_num))
                boxes_i[:, 0] = image_i
                boxes_i[:, 1:] = detection_i
                boxes.append(boxes_i)
        if len(boxes)>0:
            boxes = torch.cat(boxes, 0).to(self.device)
        else:
            boxes = torch.empty((0,8+self.class_num)).to(self.device)


        ##################
        # Generate outputs 
        ##################
        # obatin the roi score maps
        roi_score_map = self.fcn_layers(feature_map)

        # RoI cropping in a batch manner 
        cropped_img_feature = ps_roi_align(roi_score_map, boxes[:, :5], (7,7), spatial_scale=1./16)

        # combine yolo outputs and refinement vector and obtain the masks, masks[:, 0]: p(background), masks[:, 1]: p(foreground)
        regress_param, refinement_vector = self.refinement_head(cropped_img_feature)
        yolo_vector = torch.cat((boxes[:, 5:6], boxes[:, 8:]), 1)     # (obj_conf, scores_of_c_classes)
        yolo_vector.requires_grad = False
        masks = self.ensemble_head(refinement_vector, yolo_vector)

        # generate masks, i.e., the new confidence score
        positive_masks = (masks[:, 1] > self.refine_threshold)
        output = torch.cat((
            boxes[positive_masks, :1],
            box_regress(regress_param[positive_masks], boxes[positive_masks, 1:5]), 
            masks[positive_masks, 1:], 
            boxes[positive_masks, 6:8]), -1)

        # sort according to the confidence
        output = output[torch.sort(output[:,5], descending=True).indices].cpu()

        # print the grad
        """
        def save_grad():
            def hook(grad):
                print(grad)
            return hook
        # register gradient hook for masks
        if masks.requires_grad == True:
            masks.register_hook(save_grad())
        """


        ########################################
        # get labels and loss, only for training
        ########################################
        if targets is not None:

            # transform targets from xywh to x1y1x2y2; scale from (0,1) to (0,image_size)
            targets[:, 2:] = xywh2xyxy(targets[:, 2:])
            targets[:, 2:] *= images.shape[-1]   # restore to image size

            # rebuild box 
            boxes_cpu = torch.cat((boxes[:, :1], boxes[:, 7:8], boxes[:, 1:5]), 1).cpu()

            # use single process to obtain binary label for each box
            t = time.time()
            iou_labels, target_location = obtain_iou_labels(boxes_cpu, targets) # on cpu is faster than gpu
            pos_filter = (iou_labels>self.iou_thresh[1])
            neg_filter = (iou_labels<self.iou_thresh[0])

            # log infos
            conf_1 = boxes[:, 5].cpu()
            conf_2 = masks[:, 1].cpu()

            conf_1_pos, conf_2_pos = conf_1[iou_labels.flatten()>0.5], conf_2[iou_labels.flatten()>0.5]
            conf_1_neg, conf_2_neg = conf_1[iou_labels.flatten()<0.5], conf_2[iou_labels.flatten()<0.5]
            confs = dict(conf_1_pos=conf_1_pos, conf_1_neg=conf_1_neg, conf_2_pos=conf_2_pos, conf_2_neg=conf_2_neg)

            total_sample = len(iou_labels)
            refined = positive_masks.sum()
            true = pos_filter.sum()
            tps = (positive_masks.cpu() * (pos_filter.flatten())).sum().float()
            print(
                f"preocess time: {time.time()-t}",
                f"RoIs: {total_sample}", 
                f"refined_RoIs: {refined}", 
                f"true_RoIs: {true}",
                f"tps: {tps}"
                )
            metric = dict(total=total_sample, true=true, positive=refined, tp=tps, conf=confs)
            
            # balance posittive : negative = 1 : balance_fac
            pos_idx = np.where(pos_filter.flatten())[0]   # np.where(cond) returns a tuple
            neg_idx = np.where(neg_filter.flatten())[0]
            top_k = min(len(pos_idx)*self.balance_fac, len(neg_idx))

            # one-hot encoding for labels 
            label_onehot = torch.tensor([1.0, 0.0]).repeat(masks.shape[0], 1)
            for i in pos_idx:
                label_onehot[i] = torch.tensor([0.0, 1.0])   # true label: [0, 1], false label: [1, 0]

            sample_filter = pos_filter.flatten().clone()   # Tensor([True, False, ...])
            selected_neg_idx = neg_idx[random.sample(range(len(neg_idx)), k=top_k)]
            sample_filter[selected_neg_idx] = True

            label_onehot = label_onehot[sample_filter]
            masks = masks[sample_filter]     

            # focal loss for masks
            masks_loss = FocalLoss(self.device, self.alpha)(masks, label_onehot.to(self.device))
            # loss =  nn.BCELoss()(masks, label_onehot)

            # confidence loss for positive and negative samples
            conf_label = torch.zeros(len(boxes_cpu))
            for i in pos_idx:
                conf_label[i] = 1.0
            conf_loss = nn.BCELoss(reduction="sum")(
                refinement_vector[sample_filter, 0], 
                conf_label[sample_filter].to(self.device))

            # box regression loss for positive samples
            loss_xy, loss_wh = regression_loss(
                regress_param[pos_filter.flatten()], 
                target_location[pos_filter.flatten()].to(self.device), 
                boxes[pos_filter.flatten(), 1:5])

            # category loss for positive samples
            class_label = torch.zeros((len(boxes_cpu), self.class_num))
            for i, idx in enumerate(pos_idx):
                class_label[i, int(boxes_cpu[idx, 1])] = 1.0
            category_loss = nn.BCELoss(reduction="sum")(
                refinement_vector[pos_filter.flatten(), 1:], 
                class_label[pos_filter.flatten()].to(self.device))    

            loss = masks_loss + (conf_loss + category_loss)/self.loss_lambda[0] + (loss_xy + loss_wh)/self.loss_lambda[1]
            print(masks_loss.data, conf_loss.data/self.loss_lambda[0], category_loss.data/self.loss_lambda[0], \
                loss_xy.data/self.loss_lambda[1], loss_wh.data/self.loss_lambda[1])
            
        return output if targets is None else (output, loss, metric)
