import glob
import random
import os
import sys
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from utils.utils import *
from torch.utils.data import Dataset
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


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
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
        bboxs.append([items[0], int(items[1]), int(items[2]), int(items[3]), int(items[4])])
    return bboxs


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)


class ExDarkDataset(Dataset):
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
        coco_detector=False,
        img_size=416,
        augment=True,
        multiscale=False,
    ):

        # define parameters
        self.mode = mode
        self.img_size = img_size        
        self.augment = augment
        self.multiscale = multiscale
        self.coco_detector = coco_detector
    
        self.max_objects = 100
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

        self.classes = ["Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", \
            "Chair", "Cup", "Dog", "Motorbike", "People", "Table"]
        self.lighting = ["Low", "Ambient", "Object", "Single", "Weak", \
            "Strong", "Screen", "Window", "Shadow", "Twilight"]
        self.sets = ["Train", "Valid", "Test"]
        self.chosen_classes = list(range(12))   # [0,3,4,9,10]  # indices of chosen class in Exdark dataset
        self.get_paths()


    def get_paths(self):
        img_train_path, label_train_path = [], []
        img_test_path, label_test_path = [], []
        img_val_path, label_val_path = [], []

        directory_file = "../data/ExDark/imageclasslist.txt"
        file = open(directory_file, "r")
        lines = file.read().split("\n")
        lines = [x for x in lines if x and not x.startswith("#")]
        lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
        for line in lines:
            image_name, image_class, lighting, place, set_div = line.split(" ")
            if (int(image_class)-1) in self.chosen_classes:  
                if set_div in ["1", "2"]:   # train and valid
                    img_train_path.append(os.path.join("../data/ExDark/Img", self.classes[int(image_class)-1], image_name))
                    label_train_path.append(os.path.join("../data/ExDark/Label", self.classes[int(image_class)-1], image_name+".txt"))
                if set_div in ["3"]:    # test
                    img_test_path.append(os.path.join("../data/ExDark/Img", self.classes[int(image_class)-1], image_name))
                    label_test_path.append(os.path.join("../data/ExDark/Label", self.classes[int(image_class)-1], image_name+".txt"))
        
        self.paths = dict(
            train = dict(img=img_train_path, label = label_train_path), \
            valid = dict(img=img_val_path, label = label_val_path), \
            test = dict(img=img_test_path, label = label_test_path)
            )


    def __getitem__(self, idx):

        img_path = self.paths[self.mode]["img"][idx]
        label_path = self.paths[self.mode]["label"][idx]

        # ---------
        #  Image
        # ---------

        # Extract image as PyTorch tensor
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
            boxes = obtain_bboxs(label_path)
            for box in boxes:   
                if box[0] == "People":  # labels names in the ExDark annoation
                    box[0] = "person"   # label names in coco uses
                if box[0] == "Table":
                    box[0] = "diningtable"
                coco_classes = load_classes("./config/coco.names")
                box[0] = coco_classes.index(box[0].lower()) # idx of the name in the coco.names
                if self.coco_detector == False:
                    exdark_class_in_coco= [0,1,2,3,5,8,15,16,39,41,56,60]
                    box[0] = exdark_class_in_coco.index(box[0])  # class label is element in exdark_class_in_coco
            boxes = torch.tensor(boxes, dtype=float)

            # Extract coordinates for unpadded + unscaled image
            # boxes: (class_number, left, top, width, height)
            x1 = boxes[:, 1]
            y1 = boxes[:, 2]
            x2 = boxes[:, 1] + boxes[:, 3]
            y2 = boxes[:, 2] + boxes[:, 4]
            # Adjust for added padding
            x1 += pad[0]    # last dim
            x2 += pad[1]    # pad[1] == pad[0]
            y1 += pad[2]    # 2nd to last dim
            y2 += pad[3]    # pad[3] == pad[2]
            # Obtain (x_center, y_center, w, h)
            boxes[:, 1] = ((x1+x2)/2) / padded_w
            boxes[:, 2] = ((y1+y2)/2) / padded_h
            boxes[:, 3] /= padded_w
            boxes[:, 4] /= padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if torch.rand(1) < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets   


    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))    

        # Add sample index (image number in a batch) to targets
        for i, boxes in enumerate(targets):     # here target is a list with length batch_size
            if boxes is not None:
                boxes[:, 0] = i
        
        # Remove empty placeholder targets and resize to tensor(n, 6)
        targets = [boxes for boxes in targets if boxes is not None]
        targets = torch.cat(targets, 0)   
        
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape (for multiscale training)
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets


    def __len__(self):
        return len(self.paths[self.mode]["img"])
