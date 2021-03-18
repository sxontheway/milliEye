from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test_module2 import evaluate
from my_models import Network, define_yolo, init_yolo

import os
import sys
import time
import datetime
import argparse
from terminaltables import AsciiTable

import torch
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
        default=400, help="number of epochs"
        )
    parser.add_argument(
        "--batch_size", type=int, 
        default=24, help="size of each image batch"
        )
    parser.add_argument(
        "--gradient_accumulations", type=int,
        default=2, help="number of gradient accums before step"
        )
    parser.add_argument(
        "--n_cpu", type=int,
        default=24, help="number of cpu threads to use during batch generation"
        )
    parser.add_argument(
        "--img_size", type=int, 
        default=416, help="size of each image dimension"
        ) 
    parser.add_argument(
        "--multiscale_training", type=bool,
        default=True, help="allow for multi-scale training"
        )
    parser.add_argument(
        "--checkpoint_interval", type=int,
        default=1, help="interval between saving model weights"
        )
    parser.add_argument(
        "--evaluation_interval", type=int,
        default=1, help="interval evaluations on validation set"
        )

    # data params    
    parser.add_argument(
        "--conf_thresh", type=float, 
        default=0.01, help="object confidence threshold"
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
        "--checkpoint", type=str,
        help="interval between saving model weights"
        )    
    parser.add_argument(
        "--data_config",
        type=str,
        default="config/mixed.data",
        help="path to data config file",
    )

    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("checkpoints", exist_ok=True)

    # Get classes
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(opt.classes_path)

    # Initiate model
    base_detector = define_yolo(opt.yolo_cfg)
    model = Network(base_detector, opt.conf_thresh).to(device)
    if opt.checkpoint:  
        # load (yolo+module2) from checkpoint
        model.load_state_dict(torch.load(opt.checkpoint))
    else:   
        # load pre-trained yolo weights, train module2 from sketch 
        model.apply(weights_init_normal)
        init_yolo(model=model.base_detector, weights_path=opt.yolo_weights)

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    with SummaryWriter() as writer:

        for epoch in range(opt.epochs):

            model.train()
            model.base_detector.eval()

            start_time = time.time()
            for batch_i, (_, imgs, targets) in enumerate(dataloader):

                batches_done = len(dataloader) * epoch + batch_i
                epoch_batches_left = len(dataloader) - (batch_i + 1)

                imgs = imgs.to(device)
                targets.requires_grad = False

                output, loss, metric = model(imgs, targets) # img: cuda, targets: cpu
                loss.backward()

                if batches_done % opt.gradient_accumulations == 0:
                    # Accumulates gradient before each step
                    optimizer.step()
                    optimizer.zero_grad()

                # ----------------
                #   Log progress
                # ----------------
                log_str = "--- [Epoch %d/%d, Batch %d/%d] ---" % (
                    epoch, opt.epochs, batch_i, len(dataloader),
                )
                log_str += f"\nTotal loss {loss.item()}"
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

            if epoch % opt.evaluation_interval == 0:
                print("\n---- Evaluating Model ----")
                # Evaluate the model on the validation set
                precision, recall, AP, f1, ap_class, _, _ = evaluate(
                    model,
                    list_path=valid_path,
                    iou_thresh = 0.5,
                    conf_thresh = 0.01,
                    nms_thresh = 0.5,
                    img_size = opt.img_size,
                    batch_size = opt.batch_size,
                )
                evaluation_metrics = [
                    ("val_precision", precision.mean()),
                    ("val_recall", recall.mean()),
                    ("val_mAP", AP.mean()),
                    ("val_f1", f1.mean()),
                    ("val_(f1+mAP)", f1.mean()+AP.mean()),
                ]                   
                writer.add_scalars("metrics", dict(evaluation_metrics), global_step = epoch)

                # Print class APs and mAP
                ap_table = [["Index", "Class name", "AP"]]
                for i, c in enumerate(ap_class):
                    ap_table += [[c, class_names[i], "%.5f" % AP[i]]]
                print(AsciiTable(ap_table).table)
                print(f"---- mAP {AP.mean()}")

            if epoch % opt.checkpoint_interval == 0:
                if opt.yolo_weights.endswith("best_coco.pt"):
                    dataset_name = "coco"
                elif opt.yolo_weights.endswith("best_exdark.pt"):
                    dataset_name = "exdark"
                elif opt.yolo_weights.endswith("best_mixed.pt"):
                    dataset_name = "mixed"
                torch.save(model.state_dict(), f"checkpoints/ckpt_%d.pth" % epoch)
