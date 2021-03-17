from utils.utils import *
from utils.datasets import MyDataset
from utils.parse_config import *
from test_fusion import evaluate
from my_models import Network, define_yolo, init_yolo

import os
import sys
import time
import datetime
import argparse
import pickle
from terminaltables import AsciiTable

import torch, torchvision
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # Trainer params
    parser.add_argument(
        "--epochs", type=int, 
        default=100, help="number of epochs"
        )
    parser.add_argument(
        "--batch_size", type=int, 
        default=16, help="size of each image batch"
        )
    parser.add_argument(
        "--gradient_accumulations", type=int,
        default=2, help="number of gradient accums before step"
        )
    parser.add_argument(
        "--n_cpu", type=int,
        default=16, help="number of cpu threads to use during batch generation"
        )
    parser.add_argument(
        "--checkpoint_interval", type=int,
        default=1, help="interval between saving model weights"
        )
    parser.add_argument(
        "--evaluation_interval", type=int,
        default=1, help="interval evaluations on validation set"
        )
    parser.add_argument(
        "--multiscale_training", default=True, 
        help="allow for multi-scale training"
        )    

    # data params    
    parser.add_argument(
        "--conf_thresh", type=float, 
        default=0.01, help="object confidence threshold"
    )    
    parser.add_argument(
        "--img_size", type=int, 
        default=416, help="size of each image dimension"
        )
    parser.add_argument(
        "--classes_path", type=str,
        default="config/exdark.names", help="path to data config file"
        )
    parser.add_argument(
        "--yolo_cfg", type=str,
        default="config/yolov3-tiny-12.cfg", help="train and test mode"
        )
    parser.add_argument(
        "--yolo_weights", type=str,
        default="weights/best_mixed.pt", help="train and test mode"
        ) 
    parser.add_argument(
        "--pretrained_module2", type=str,
        default="./weights/module2_best_mixed.pth", help="train and test mode"
        )
    parser.add_argument(
        "--checkpoint", type=str,
        help="interval between saving model weights"
        )    
    parser.add_argument(
        "--illumination", type=str,
        default=['H','L'], help="train and test mode"
        )
    parser.add_argument(
        "--test_list", type=int,
        default=4, help="test scenem: 0, 1, 2, 3, 4"
        )
        
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("checkpoints", exist_ok=True)

    # Get classes
    class_names = load_classes(opt.classes_path)

    # Initiate model
    base_detector = define_yolo(opt.yolo_cfg)
    model = Network(base_detector, opt.conf_thresh).to(device)

    if opt.checkpoint:  
        # load (yolo+module2+module3) from checkpoint
        model.load_state_dict(torch.load(opt.checkpoint))
    else:   
        # load pre-trained yolo weights, train (module2+module3) from sketch 
        model.apply(weights_init_normal)
        init_yolo(model=model.base_detector, weights_path=opt.yolo_weights)

    if opt.pretrained_module2:
        param = torch.load(opt.pretrained_module2)
        para_name = list(param)             # param: a collections.OrderedDict
        module_list = model.state_dict()    # module_list: a collections.OrderedDict

        names1 = ["img_cnn_layers.net.conv_0.weight", "img_cnn_layers.net.conv_0.bias", 
        "img_cnn_layers.net.batch_norm_0.weight", "img_cnn_layers.net.batch_norm_0.bias", "img_cnn_layers.net.batch_norm_0.running_mean", 
        "img_cnn_layers.net.batch_norm_0.running_var", "img_cnn_layers.net.batch_norm_0.num_batches_tracked", 
        "refinement_head.net0.0.weight", "refinement_head.net0.0.bias", 
        "refinement_head.net1.0.weight", "refinement_head.net1.0.bias", 
        "refinement_head.net2.0.weight", "refinement_head.net2.0.bias"]
        names2 = ["fcn_layers.net.conv_0.weight", "fcn_layers.net.conv_0.bias", 
        "fcn_layers.net.batch_norm_0.weight", "fcn_layers.net.batch_norm_0.bias", "fcn_layers.net.batch_norm_0.running_mean",
        "fcn_layers.net.batch_norm_0.running_var", "fcn_layers.net.batch_norm_0.num_batches_tracked",
        "refinement_head.net0.0.weight", "refinement_head.net0.0.bias", 
        "refinement_head.net1.0.weight", "refinement_head.net1.0.bias", 
        "refinement_head.net2.0.weight", "refinement_head.net2.0.bias"]

        # generate an OrderedDict
        tmp = []
        for name in para_name:
            if name in names2:
                tmp.append(param[name])
        for name in module_list:
            if name in names1:
                module_list[name] = tmp.pop(0)
        model.load_state_dict(module_list)

        # freeze some layers 
        for name, param in model.named_parameters():
            if name in names1:
                print(name)
                param.requires_grad = False
    

    # Get dataloader
    dataset = MyDataset(mode="train", illumination=opt.illumination, augment=False, multiscale=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    with SummaryWriter() as writer:

        for epoch in range(opt.epochs):

            model.train()
            model.base_detector.eval()

            start_time = time.time()
            for batch_i, (_, imgs, targets, radar_boxes, radar_maps) in enumerate(dataloader):
        
                batches_done = len(dataloader) * epoch + batch_i
                epoch_batches_left = len(dataloader) - (batch_i + 1)

                # ---------------
                # backpropagation
                # ---------------
                imgs = imgs.to(device)  # to gpu if any
                radar_maps = radar_maps.to(device) 
                radar_boxes = radar_boxes.to(device)

                loss, outputs, metric, radar_attention = model(imgs, radar_maps, radar_boxes, targets.clone())
                loss.backward()

                if batches_done % opt.gradient_accumulations == 0:
                    # Accumulates gradient before each step
                    optimizer.step()
                    optimizer.zero_grad()

                # ---------------------------------------------
                # inspect data from dataload using tenosorboard
                # ---------------------------------------------
                if batches_done % 50 ==0:  
                    # radar maps
                    grid = torchvision.utils.make_grid(radar_maps, pad_value = 0.5)
                    writer.add_image('maps', grid, global_step = batches_done)

                    # images
                    writer.add_images('image', imgs, global_step = batches_done)

                    # radar attention maps
                    writer.add_images('radar_attention', radar_attention[:,:3], global_step = batches_done)

                    # targets: [0, class_num, x_center, y_center, w, h]
                    """
                    # show in tensorboard
                    size = imgs.shape[2]
                    targets_tmp = targets.clone()
                    targets_tmp = targets_tmp[targets_tmp[:,0]==0]    
                    targets_tmp = targets_tmp[:, 2:]*size
                    targets_tmp = xywh2xyxy(targets_tmp)        
                    writer.add_image_with_boxes('img_with_boxes', imgs[0], targets_tmp, global_step = batches_done)
                    """
                    size = imgs.shape[2]
                    radar_boxes_tmp = radar_boxes.clone()
                    radar_boxes_tmp = radar_boxes_tmp[radar_boxes_tmp[:,0]==0]           
                    writer.add_image_with_boxes('img_with_boxes', imgs[0], radar_boxes_tmp, global_step = batches_done)
                    

                # -------------
                # Log progress
                # -------------
                log_str = "--- [Epoch %d/%d, Batch %d/%d] ---" % (
                    epoch, opt.epochs, batch_i, len(dataloader),
                )
                log_str += f"\nTotal loss: {loss.item()}"
                # Determine approximate time left for epoch
                time_left = datetime.timedelta(
                    seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1)
                )
                log_str += f"\n---- ETA {time_left}\n"
                print(log_str)

                writer.add_scalar("loss", loss, global_step = batches_done)
                writer.add_scalar("precesion", metric["tp"]/metric["positive"], global_step = batches_done)     
                writer.add_scalar("recall", metric["tp"]/metric["true"], global_step = batches_done)  
                model.seen += imgs.size(0)

            if epoch % opt.checkpoint_interval == 0:
                torch.save(model.state_dict(), f"./checkpoints/{opt.test_list}_ckpt_{epoch}.pth")
                
            if epoch % opt.evaluation_interval == 0:
                print("\n---- Evaluating Model ----")
                # Evaluate the model on the validation set
                precision, recall, AP, f1, ap_class, _, _ = evaluate(
                    model,
                    mode = "test",
                    model_mode = 0, 
                    illumination = ["L"],   # opt.illumination,
                    iou_thresh = 0.5,
                    nms_thresh = 0.5,
                    img_size = opt.img_size,
                    batch_size = opt.batch_size,
                    test_list = opt.test_list
                )
                evaluation_metrics = [
                    ("val_precision", precision.mean()),
                    ("val_recall", recall.mean()),
                    ("val_mAP", AP.mean()),
                    ("val_f1", f1.mean())
                ]                   
                writer.add_scalars("metrics", dict(evaluation_metrics), global_step = epoch)

                # Print class APs and mAP
                ap_table = [["Index", "Class name", "AP"]]
                for i, c in enumerate(ap_class):
                    ap_table += [[c, class_names[i], "%.5f" % AP[i]]]
                print(AsciiTable(ap_table).table)
                print(f"---- mAP {AP.mean()}")
