import numpy as np
import random
import torch
from torch import nn
from torchvision.ops import ps_roi_align, roi_align

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


class cnn_layers_1(nn.Module):
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
        super(cnn_layers_1, self).__init__()
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


class modailty_reweight(nn.Module):
    """
    Reweight importance of two sensors
    """
    def __init__(self):
        super(modailty_reweight, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(490, 2), 
            nn.Sigmoid(),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = nn.AvgPool2d((x.shape[2:4]))(x)            # output: n*490*1*1
        x = self.net(x.squeeze(-1).squeeze(-1))     # output: n*2
        return x


class cnn_layers_2(nn.Module):
    def __init__(self):
        super(cnn_layers_2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, momentum=0.1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )   # 32*32*3 -> 16*16*32
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.LeakyReLU(0.1)
            )   # 16*16*32 -> 16*16*64
        
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, momentum=0.1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 1, kernel_size=1, stride=1)
            )   # 16*16*64 -> 32*32*32 -> 32*32*32 -> 32*32*1
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        out = nn.Sigmoid()(x)
        return out


class cnn_layers_3(nn.Module):
    def __init__(self):
        super(cnn_layers_3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, momentum=0.1),
            nn.LeakyReLU(0.1),
            )   # 32*32*3 -> 32*32*32
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.LeakyReLU(0.1)
            )   # 32*32*32 -> 32*32*64
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 10, kernel_size=1, stride=1),
            )   # 32*32*64 -> 32*32*128 -> 32*32*1
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        out = nn.Sigmoid()(x)
        return out


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

    
class ensemble_head(nn.Module):
    """
    Fuse two feature vectors using attention-based fusion
    ---
    params for __init__():
        -channels (list): number of neuros in each layer, e.g. (2, 32, 32, 2)
        -use_activation: if there is a softmax during the calculation of loss function,  
        then this should be set to False 
    ---
    forward():
        -inputs: two vectors. e.g., tensor(n, c+1), tensor(n, c+1)
        -outputs: mask vector, e.g. tensor(n*2), for (p_foreground, p_background)
    """

    def __init__(self, channels, activation_softmax=True):
        super(ensemble_head, self).__init__()
        self.activation_softmax = activation_softmax
        self.fc1 = nn.Sequential(
            nn.Linear(channels[0], channels[1]),
            nn.LeakyReLU(0.1)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(channels[2], channels[3]),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, refinement_vector, yolo_vector):
        x = torch.stack((refinement_vector, yolo_vector), -1)     # n*(c+1)*channels[0]
        x = self.fc1(x)     # n*(c+1)*channels[1]
        x = x.flatten(start_dim=1)      # n*channel[2]
        x = self.fc2(x)     # n*channel[3]
        if self.activation_softmax == True:
            x = self.softmax(x)

        return x


class refinement_head(nn.Module):
    """
    Fuse two feature vectors using attention-based fusion
    ---
    params for __init__():
        -channels (list): number of neuros in each layer, e.g. (490, 256, 128, c+1)
        -use_activation: if there is a softmax during the calculation of loss function,  
        then this should be set to False 
    ---
    forward():
        -inputs: two feature maps, e.g. tensor(n*10*7*7)
        -outputs: feature vector, e.g. tensor(n*2)
    """

    def __init__(self, channels):
        super(refinement_head, self).__init__()
        self.count = 0

        tmp = 49

        self.net0 = nn.Sequential(
            nn.Linear(channels[0], channels[1]),    # 490->256
            nn.LeakyReLU(0.1)
        )
        self.net1 = nn.Sequential(
            nn.Linear(channels[1], 4),    # 256->4
        )
        self.net2 = nn.Sequential(
            nn.Linear(channels[1], 13),    # 256->c+1
            nn.Sigmoid()
        )
        self.net3 = nn.Sequential(
            nn.Linear(channels[1], tmp),    # 256->tmp
            nn.Sigmoid(),
        )   
        self.radar_net = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(10, momentum=0.1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(10, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(2*tmp, 1),
            nn.Sigmoid()
        )

    def forward(self, radar_maps, img_maps):

        img_maps_flatten = img_maps.flatten(start_dim=1)
        img_tmp = self.net0(img_maps_flatten)
        box_regression = self.net1(img_tmp)
        class_vector = self.net2(img_tmp)

        # radar_maps_flatten = radar_maps.flatten(start_dim=1)
        radar_confidence = self.radar_net(radar_maps).squeeze(-1).squeeze(-1)   # n*1
        confidence = radar_confidence + class_vector[:, :1]
        confidence = nn.Sigmoid()(confidence)

        # with open('a.txt', 'a+') as f:
        #     lenth = len(radar_confidence) 
        #     if lenth == 0:
        #         f.writelines(str(0) + " " + str(0) + " " + str(0) + " " + str(self.count) + "\n")
        #     else: 
        #         for (i,j,k) in zip(radar_confidence, class_vector[:, :1], confidence):
        #             f.writelines(str(i.squeeze().cpu().numpy()) + " " + str(j.squeeze().cpu().numpy()) + " " + str(k.squeeze().cpu().numpy()) + " " + str(self.count) + "\n")
            
        self.count += 1

        refinement_vector = torch.cat((confidence, class_vector[:, 1:2]), -1)

        return box_regression, refinement_vector


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
            with open('b.txt', 'a') as f:
                f.writelines("0" + "\n")
            continue
            
        # If all targets are found, break ealier
        # if multi_boxes is False and len(detected_boxes) == len(targets):
        #     break   
        
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

            with open('b.txt', 'a') as f:
                if iou>0.5:
                    f.writelines("1" + "\n")
                else:
                    f.writelines("0" + "\n")

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
        self.balance_factor = 5
        self.loss_lambda = (6, 1)
        self.refine_threshold_img, self.refine_threshold_radar = 0, 0
        self.class_num = 1
        self.class_idx = 0

        self.base_detector = base_detector.eval()
        self.img_cnn_layers = cnn_layers_1((256, 490))
        self.radar_cnn_layers = cnn_layers_3()  # (3, 32, 64, 128, 1)

        self.refinement_head = refinement_head((490, 256, 128, self.class_num+1))
        self.ensemble_head = ensemble_head((2, 32, 32*(1+self.class_num), 2))

    def forward(self, images, maps, radar_boxes_location, model_mode=0, targets=None):
        """
        Arguments:
            -images: Tensor[N,C,H,W], here H=W=416
            -maps: Tensor [N,C,H',W'], here H'=W'=32
            -radar_boxes_location: Tensor[n, 5], 5 for (index_in_a_batch, x, y, x ,y)
            -model_mode: 0, 1, 2, 3 for [millieye, yolo, radar, auto]
            -targets: Tensor[m, 6], m is the number of annotated bboxes in a batch.  
            6 for [index_in_a_batch, class_num, x_center, y_center, w, h],
            Here (x_center, y_center, w, h) is scale to (0, 1)

        Returns:
            -loss: float
            -output: the new score for every bbox, Tensor[m, 8].
                8 for [image_i, x1, y1, x2, y2, object_conf, class_score, class_pred]
                Here (x1, y1, x2, y2) is scale to the size of images (e.g., 416)
        """
        ########################################
        # Get candidate boxes from base detector
        ########################################
        # for tiny yolov3, feature_map: (1,256,26,26); output_tensor: (1,2535,85). No gradient.
        feature_map, output_tensor = self.base_detector(images)   

        # NMS in a batch manner, CPP version
        detections = non_max_suppression_cpp(output_tensor.clone().cpu(), conf_thresh = self.conf_thresh)

        # yolo detections: List[tensor[n,7+c]] -> img_boxes: tensor[N,8+c]
        img_boxes = []
        for image_i, detection_i in enumerate(detections):  # iter through a batch
            if detection_i is not None:
                detection_i = detection_i[detection_i[:, 6] == self.class_idx]  # only keep certain classes
                if len(detection_i)>0:
                    boxes_i = torch.zeros((len(detection_i), 8+self.class_num))
                    boxes_i[:, 0] = image_i
                    boxes_i[:, 1:] = detection_i[:, :7+self.class_num]
                    img_boxes.append(boxes_i)
        if len(img_boxes)>0:
            img_boxes = torch.cat(img_boxes, 0).to(self.device)
        else:
            img_boxes = torch.empty((0, 8+self.class_num)).to(self.device)
        num_img_boxes = len(img_boxes)

        # yolo only mode 
        if model_mode == 1:      
            return img_boxes[: ,:8]
        # radar only mode
        if model_mode == 2:
            self.refine_threshold_img = 1

        ##################
        # Generate outputs
        ##################
        # obatin the roi score maps
        roi_score_map = self.img_cnn_layers(feature_map)    # back propagation stops at feature_map
        radar_score_map = self.radar_cnn_layers(maps)#.sum(1).unsqueeze(1)    # back propagation stops at map

        # aggregate RoI proposals from camera and radar, crop in a batch manner 
        if len(radar_boxes_location)>0:
            radar_boxes_location[:, 1:] *= images.shape[-1]
        box_locations = torch.cat((img_boxes[:, :5], radar_boxes_location), 0)    # scaled to (0, image_size)

        # RoI cropping
        cropped_image_feature = ps_roi_align(roi_score_map, box_locations, (7,7), spatial_scale=1./16)
        cropped_radar_feature = roi_align(radar_score_map, box_locations, (7,7), spatial_scale=1./16)

        # combine yolo outputs and refinement vector and obtain the masks
        regress_param, refinement_vector = self.refinement_head(cropped_radar_feature, cropped_image_feature)   # n*4, n*(1+c)

        # aggregate raw img proposals from yolo + refined radar proposals (box locations are not regressed)
        radar_boxes = torch.cat((
            radar_boxes_location,   # n*5, for idx in a batch and location
            refinement_vector[num_img_boxes:],  # box confidience, the highest class score
            torch.zeros((len(radar_boxes_location), 1)).to(self.device),    # class predicted
            refinement_vector[num_img_boxes:, 1:],    # class scores for c categories
            ), -1)
        boxes = torch.cat((img_boxes, radar_boxes), 0)
        yolo_vector = torch.cat((img_boxes[:, 5:6], img_boxes[:, 8:]), 1).detach()     # n*(1+c)

        # generate masks, masks[:, 0]: p(background), masks[:, 1]: p(foreground)
        masks_img_proposals = self.ensemble_head(refinement_vector[:num_img_boxes], yolo_vector) # input n*(1+c). n*(1+c), output n*2
        masks = torch.cat((masks_img_proposals[:, :1], refinement_vector[num_img_boxes:, :1]), 0)   # n*1
        masks = torch.cat((1-masks, masks), -1)   # n*2

        # repalce old confidence scores with new ones
        positive_masks = torch.cat(
            (masks[:num_img_boxes, 1] > self.refine_threshold_img, 
            masks[num_img_boxes:, 1] > self.refine_threshold_radar),
            0).cpu()
        
        # not radar_only mode
        if model_mode != 2: 
            output = torch.cat((
                boxes[positive_masks, :1],
                box_regress(regress_param[positive_masks], boxes[positive_masks, 1:5]), 
                masks[positive_masks, 1:],  # box confidence
                boxes[positive_masks, 6:8]), -1)    # class score, class predict
        else:   # radar only mode, skip position regression
            output = torch.cat((
                boxes[positive_masks, :1],
                boxes[positive_masks, 1:5], 
                masks[positive_masks, 1:],  # box confidence
                boxes[positive_masks, 6:8]), -1)    # class score, class predict  

        # sort according to the confidence
        masks_tmp = masks.clone()
        masks_tmp[num_img_boxes:, 1] /= 5   # set high priority for camera proposals
        output = output[torch.sort(masks_tmp[positive_masks,1], descending=True).indices]


        ########################################
        # get labels and loss, only for training
        ########################################
        if targets is not None:

            # transform targets from xywh to x1y1x2y2; scale from (0,1) to (0,image_size)
            targets[:, 2:] = xywh2xyxy(targets[:, 2:])
            targets[:, 2:] *= images.shape[3]   # restore to image size

            # rebuild box 
            boxes_cpu = torch.cat((boxes[:, :1], boxes[:, 7:8], boxes[:, 1:5]), 1).cpu()

            # use single process to obtain binary label for each box
            t = time.time()
            iou_labels, target_location = obtain_iou_labels(boxes_cpu, targets, self.iou_thresh)  # on cpu
            pos_filter = (iou_labels>self.iou_thresh[1])
            neg_filter = (iou_labels<self.iou_thresh[0])

            ###########
            # log infos
            ###########
            conf_1 = boxes[:, 5].cpu()
            conf_2 = masks[:, 1].cpu()

            conf_1_pos, conf_2_pos = conf_1[iou_labels.flatten()>0.5], conf_2[iou_labels.flatten()>0.5]
            conf_1_neg, conf_2_neg = conf_1[iou_labels.flatten()<0.5], conf_2[iou_labels.flatten()<0.5]
            confs = dict(conf_1_pos=conf_1_pos, conf_1_neg=conf_1_neg, conf_2_pos=conf_2_pos, conf_2_neg=conf_2_neg)

            total_sample = len(iou_labels)
            true = pos_filter.sum()
            positive = positive_masks.sum()
            tps = (positive_masks * (pos_filter.flatten())).sum().float()
            print(
                f"preocess time: {time.time()-t}",
                f"total_samples: {total_sample}", 
                f"positive_masks: {positive}", 
                f"true samples: {true}",
                f"tps: {tps}"
                )
            metric = dict(total=total_sample, true=true, positive=positive, tp=tps, conf=confs)

            # check img and radar proposal pos/neg distributions
            print("***", pos_filter[:num_img_boxes].sum(), pos_filter[num_img_boxes:].sum() )
            print("###", neg_filter[:num_img_boxes].sum(), neg_filter[num_img_boxes:].sum() ) 

            ###################################################
            # sample balancing for both img and radar proposals
            ###################################################
            pos_idx = np.where(pos_filter.flatten())[0]   # np.where(cond) returns a tuple
            neg_idx = np.where(neg_filter.flatten())[0]
            top_k = min(len(pos_idx)*self.balance_factor, len(neg_idx))

            # one-hot encoding for labels 
            label_onehot = torch.tensor([1.0, 0.0]).repeat(masks.shape[0], 1)
            for i in pos_idx:
                label_onehot[i] = torch.tensor([0.0, 1.0])   # true label: [0, 1], false label: [1, 0]

            sample_filter = pos_filter.flatten().clone()   # Tensor([True, False, ...])
            selected_neg_idx = neg_idx[random.sample(range(len(neg_idx)), k=top_k)]
            sample_filter[selected_neg_idx] = True

            label_onehot = label_onehot[:num_img_boxes][sample_filter[:num_img_boxes]]
            onehot_masks = masks[:num_img_boxes][sample_filter[:num_img_boxes]]

            ##################
            # loss calculating
            ##################
            # focal loss for image proposals only
            masks_loss = FocalLoss(self.device, self.alpha)(onehot_masks, label_onehot.to(self.device))
            # loss =  nn.BCELoss()(onehot_masks, label_onehot)

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

            loss = masks_loss + (conf_loss)/self.loss_lambda[0] # + (loss_xy + loss_wh)/self.loss_lambda[1]
            print(masks_loss.data, conf_loss.data/self.loss_lambda[0], category_loss.data/self.loss_lambda[0], \
                loss_xy.data/self.loss_lambda[1], loss_wh.data/self.loss_lambda[1])
  
            radar_attention = radar_score_map[:,:1,:,:].detach()  # n*c*w*h
            
        return output if targets is None else (loss, output, metric, radar_attention)