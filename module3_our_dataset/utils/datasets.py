import pickle
import random
import os
import sys
import time
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from utils.utils import *
from torch.utils.data import Dataset, IterableDataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size, mode="nearest"):
    image = F.interpolate(image.unsqueeze(0), size=size, mode=mode).squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448, mode="nearest"):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode=mode)
    return images


def obtain_bboxs(path) -> list:
    """
    obatin bbox annotations from the file
    """
    file = open(path, "r")
    lines = file.read().split("\n")
    lines = [x for x in lines if x and not x.startswith("%")]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces

    bboxs = []
    for line in lines:
        items = line.split(" ")
        bboxs.append([items[0], float(items[1]), float(items[2]), float(items[3]), float(items[4])])
    return bboxs


def plot_radar_heatmap(points, img_size, radar_maps_size=32, save_path=None):
    """
    plot the radar heatmap and save (optional). If do not input `save_path`, then no heatmap will be saved.
    ---
    params
    ---
        -points: in image coordinate, points[0, :] for width(right), points[1, :] for height(down)
        -img_size: the original PIL image size: width * height, e.g. 1600 * 900 
    return
    ---
        -the radar maps: np.array WHC, e.g., (32,24,3)
    ---
    """
    scale = max(img_size)/radar_maps_size
    bin_w, bin_h = round(img_size[0]/scale), round(img_size[1]/scale)

    # point counting heatmap: the denser, the bigger
    h0 = np.histogram2d(x=points[0, :], y=points[1, :], bins=[bin_w, bin_h],\
        range=[[0, img_size[0]],[0, img_size[1]]])[0].T

    # depth heatmap: the near the bigger
    h1 = np.histogram2d(x=points[0, :], y=points[1, :], bins=[bin_w, bin_h], \
        range=[[0, img_size[0]],[0, img_size[1]]], weights=points[2, :])[0].T
    h1 /= (h0+1e-6)     # mean value in each bin
    h1 = np.where(h1<1, 100, h1)   # if depth=0, change to 100 (>100 will be ignore later) 

    # absolute v maps: v_x or v_y: the faster. the bigger
    h2 = np.histogram2d(x=points[0, :], y=points[1, :], bins=[bin_w, bin_h], \
        range=[[0, img_size[0]],[0, img_size[1]]], weights=points[3, :])[0].T
    h2 = np.absolute(h2/(h0+1e-6))  # mean value in each bin

    # print(h0.max().max(), h1.max().max(), h2.max().max())
    maps = np.stack((h0, h1, h2), axis=-1)  # with shape (bin_h, bin_w, c), e.g.(32,18,3)

    # rescale to (0,1)
    ranges = ((0,5),(12,0),(0,4))  # range for count, depth, v channel respectively
    for i in range(maps.shape[-1]):
        h = maps[...,i]
        h_min, h_max = ranges[i]
        maps[...,i] = np.clip((h-h_min)/(h_max-h_min), 0, 1)

    # save plot (optional)
    save_path =  None#f"./show/a_{time.time()}.jpg"
    if save_path is not None:
        plt.imshow(maps, vmin=0, vmax=1, cmap='jet') 
        plt.axis('on')
        plt.colorbar()
        plt.savefig(save_path)
    plt.close('all')

    return maps


class MyDataset(Dataset):
    """
    The return of __getitem__(): (img_path, img, targets)
    ---
        -img_path: str  
        -img: the padded image, tensor[3, 416, 416]  
        -targets: tensor[k, 6], 6 is for [0, class_num, x_center, y_center, w, h]. 
        (x_center, y_center, w, h) is normalized into (0, 1) according to the padded size

    The return of collate_fn(): (paths, imgs, targets)
    ---
        -targets: tensor[k, 6], k is the total number of labeled bboxes in a batch
        the first element of every entry denotes the image index in a batch (starting with 0)
    """

    def __init__(
        self,
        mode,
        illumination,
        img_size=416,
        augment=False,
        multiscale=True,
        test_list=0,
        dataset_folder="../data/our_dataset"
    ):

        # define parameters
        self.mode = mode
        self.illumination = illumination
        self.img_size = img_size   
        self.map_size = int(self.img_size/16)       
        self.augment = augment
        self.multiscale = multiscale
        self.test_list = ["0", "1", "2", "3", "4"][test_list:test_list+1] 
        self.train_list = ["0", "1", "2", "3", "4"][:test_list] + ["0", "1", "2", "3", "4"][test_list+1:]
    
        self.max_objects = 100
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

        self.chosen_classes = list(range(12))   # [0,3,4,9,10]  # indices of chosen class in Exdark dataset
        self.get_paths(dataset_folder)


    def get_paths(self, dataset_folder):

        img_train, label_train, box_train, point_train = [], [], [], []
        img_test, label_test, box_test, point_test = [], [], [], []

        directory_file = f"{dataset_folder}/dataset.txt"
        file = open(directory_file, "r")
        lines = file.read().split("\n")
        lines = [x for x in lines if x and not x.startswith("#")]
        lines = [x.rstrip().lstrip() for x in lines]
        for line in lines:
            light = line.split("-")[0][0]
            scene = line.split("-")[0][1]
            appendix = line.split("-")[2]
            low_light_list = ["153937", "211008", "211738", "212944", "213410", "213435"]   # 0-1, 1-1, 1-2, 3-1, 3-2

            # only use for testing dark scenarios
            # dark_list = ["191246", "212140", "212944" , "213208", "213242"]    # 0-2, 2-1, 2-2, 4-1, 4-2
            # if light == "L" and appendix in dark_list:
            #     light = "D" # dark is 4 f-fold
                
            if light in self.illumination:
                if scene in self.train_list:   
                    img_train.append(os.path.join(f"{dataset_folder}/image", line+".jpg"))
                    label_train.append(os.path.join(f"{dataset_folder}/label", line+".txt"))
                    box_train.append(os.path.join(f"{dataset_folder}/radar_box", line+".pkl"))
                    point_train.append(os.path.join(f"{dataset_folder}/radar_point", line+".pkl"))
                if scene in self.test_list:  # 5-fold
                    img_test.append(os.path.join(f"{dataset_folder}/image", line+".jpg"))
                    label_test.append(os.path.join(f"{dataset_folder}/label", line+".txt"))
                    box_test.append(os.path.join(f"{dataset_folder}/radar_box", line+".pkl"))
                    point_test.append(os.path.join(f"{dataset_folder}/radar_point", line+".pkl"))

        self.paths = dict(
            train = dict(img=img_train, label=label_train, box=box_train, point=point_train), \
            test = dict(img=img_test, label=label_test, box=box_test, point=point_test)
            )


    def __getitem__(self, idx):

        img_path = self.paths[self.mode]["img"][idx]
        label_path = self.paths[self.mode]["label"][idx]
        box_path = self.paths[self.mode]["box"][idx]
        point_path = self.paths[self.mode]["point"][idx]

        # ---------
        #  Image
        # ---------
        save_path = None#f"./show/b_{time.time()}.jpg"
        if save_path is not None:
            plt.imshow(Image.open(img_path)) 
            plt.axis('on')
            plt.savefig(save_path)
        plt.close('all')
        img = transforms.ToTensor()(Image.open(img_path).convert("RGB"))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        # image padding
        _, h, w = img.shape     # (h, w) is the original size 
        img, pad = pad_to_square(img, 0)    
        _, padded_h, padded_w = img.shape   # img.shape is the padded size

        # ---------
        #  Label
        # ---------
        targets = None
        if os.path.exists(label_path):  
            # boxes: (class_number, left, top, width, height)
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5)) # xywh

            # Extract coordinates for unpadded + unscaled image
            x1 = (boxes[:, 1] - boxes[:, 3] / 2)*w
            y1 = (boxes[:, 2] - boxes[:, 4] / 2)*h
            x2 = (boxes[:, 1] + boxes[:, 3] / 2)*w
            y2 = (boxes[:, 2] + boxes[:, 4] / 2)*h
            # Adjust for added padding
            x1 += pad[0]    # last dim
            x2 += pad[1]    # pad[1] == pad[0]
            y1 += pad[2]    # 2nd to last dim
            y2 += pad[3]    # pad[3] == pad[2]
            # Obtain (x_center, y_center, w, h), scale to (0, 1)
            boxes[:, 1] = ((x1+x2)/2) / padded_w
            boxes[:, 2] = ((y1+y2)/2) / padded_h
            boxes[:, 3] *= w/padded_w
            boxes[:, 4] *= h/padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes   

        # ---------
        #  Box
        # ---------
        with open(box_path, 'rb') as handle:
            radar_box = torch.from_numpy(pickle.load(handle))   # numpy array n*4, xyxy, scaled to image size

        radar_box_output = None
        if len(radar_box) > 0:
            # Adjust for added padding
            radar_box[:, 0] += pad[0]    # last dim
            radar_box[:, 2] += pad[1]    # pad[1] == pad[0]
            radar_box[:, 1] += pad[2]    # 2nd to last dim
            radar_box[:, 3] += pad[3]    # pad[3] == pad[2]
            # radar_box = xyxy2xywh(radar_box) 
            
            # scale to (0, 1) 
            radar_box = torch.clamp(radar_box/padded_h, 0, 1)  # padded_h == padded_w
            radar_box = radar_box[torch.logical_and(radar_box[:, 0]<radar_box[:, 2], radar_box[:, 1]<radar_box[:, 3])]
            if len(radar_box)>0:
                radar_box_output = torch.zeros((len(radar_box), 5))
                radar_box_output[:, 1:] = radar_box 
        
        #  ---------
        #  Point
        # ---------
        with open(point_path, 'rb') as handle:
            points = pickle.load(handle)  # numpy array n*4, uvzV
            radar_map = transforms.ToTensor()(plot_radar_heatmap(points.transpose(), (w, h))).float()

        # radar map padding
        radar_map, pad = pad_to_square(radar_map, 0)    
      
        # Apply augmentations
        # if self.augment:
        #     if torch.rand(1) < 0.5:
        #         img, targets = horisontal_flip(img, targets)

        return img_path, img, targets, radar_box_output, radar_map 


    def collate_fn(self, batch):
        paths, imgs, targets, radar_boxes, radar_maps = list(zip(*batch))    

        # Add sample index (image number in a batch) to targets
        for i, boxes in enumerate(targets):     # here target is a list with length batch_size
            if boxes is not None:
                boxes[:, 0] = i
        for i, boxes in enumerate(radar_boxes): 
            if boxes is not None:
                boxes[:, 0] = i
        
        # Remove empty placeholder targets and resize to tensor(n, 6)
        targets = [boxes for boxes in targets if boxes is not None]  
        if len(targets)>0:          
            targets = torch.cat(targets, 0) 
        else:
            targets = torch.empty(0,6)

        
        radar_boxes = [box for box in radar_boxes if box is not None]
        if len(radar_boxes)>0:
            radar_boxes = torch.cat(radar_boxes, 0) 
        else:
            radar_boxes = torch.empty(0,5)
        
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
            self.map_size = int(self.img_size/16)

        # Resize images to input shape (for multiscale training)
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        radar_maps = torch.stack(
            [F.interpolate(radar_map.unsqueeze(0), self.map_size, mode='bilinear', align_corners=True).squeeze(0) 
            for radar_map in radar_maps]
            )
        self.batch_count += 1

        return paths, imgs, targets, radar_boxes, radar_maps


    def __len__(self):
        return len(self.paths[self.mode]["img"])