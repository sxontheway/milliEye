# milliEye
This is the repo for IoTDI 2021 paper: "milliEye: A Lightweight mmWave Radar and Camera Fusion System for Robust Object Detection".

<br>

# Requirements
The following environment has been tested: 
* Ubuntu 18.04
* Python 3.6.8
* Pytorch 1.5.0 
* torchvision 0.6.0
* numpy 1.18.2
* tensorboardX 2.0
* opencv-python 4.5.1.48


<br>


# Strcuture
```
|-- data_collection         // data collection, showing and preparation
|-- data                    // folder to store data
    |-- coco
    |-- ExDark
    |-- mixed
    |-- our_dataset
|-- module2_mixed          // folder for the second stage training (on mixed dataset)
|-- module3_our_dataset    // folder for the third stage training (on our_dataset)
|-- pictures               // figures for this document
```

<br>

# Run
<p align="center" >
	<img src="./pictures/milliEye.png" width="1000">
</p>

## Data Preparation:
* Hardware
    * Common USB2.0 camera
    * Texas Instrument IWR6843ISK ES1.0 (ES2.0 is also supported by the script)
* Software: see the `./data_collection/README.md` for details 

## Train Yolo Tiny v3 on mixed dataset of COCO and ExDark
* Dataset downloading
    * COCO train_val 2014: https://cocodataset.org/#home 
    * ExDark: https://github.com/cs-chan/Exclusively-Dark-Image-Dataset 
* Dataset preparing and training
    * First need to transform ExDark dataset into COCO's format, please follow the instruction at: https://github.com/ultralytics/yolov3  
* Our trained model can be available at: [Onedrive](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155118647_link_cuhk_edu_hk/EpkycykJuuJAiKr_9plZ3HoB3-s9_GPmSUX-wFrHfjc_hg?e=KGKzsc) 

## Second-stage Training: image-branch + R-CNN
xx

## Third-stage Training: other radar-related parts
xx

<br>

# Citation
If you find this work useful for your research, please cite:
``` 
"Xian Shuai, Yulin Shen, Yi Tang, Shuyao Shi, Luping Ji, and Guoliang Xing, milliEye: A Lightweight mmWave Radar and Camera Fusion System for Robust Object Detection. In Proceedings of Internet of Things Design and Implementation (IoTDI’21)."
```
The Bibtex will come later. 

# Demo
See `./pictures/indoor.gif`
<p align="center" >
	<img src="./pictures/indoor.gif" width="1000">
</p>



