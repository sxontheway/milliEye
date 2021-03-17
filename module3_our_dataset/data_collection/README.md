## Files:
* `collect.py`: collect data
* `prepare_data.py`: show collected data and generate dataset
* `realtime_show.py`: show fused data in real-time


## Folders:

* `utils`: function libraries
* `cfg`: mmWave radar configuration files
    * For ES1.0 mmWave radar device, use `indoor.cfg`
    * For ES1.0 mmWave radar device, use `indoor_ES2.0.cfg`
* `yaml`: mmWave radar to camera projection
    * For customized carrier board for radar and camera, please first do calibration and update the calibration file. I use ROS camera calibration toolbox: http://wiki.ros.org/camera_calibration 
* `data`: named by created time `yyyymmdd-hhmmss`. In each `yyyymmdd-hhmmss`, there are three separate files:   
    * `video.mp4`: the video file from camera
    * `pointcloud.pkl`: the point cloud file from radar
    * `timestamps.txt`: the time stamp of each frame when capturing the video  

    A sample snippet is provided here. 

*   `dataset`: after running script `prepare_data.py`, dataset used for training will be automatically generated. It will include four folders and one `.txt` file
    * radar_point/
    * radar_box/
    * image/
    * label/
    * dataset.txt

## Debugs 
if cannot open the radar to collect data: 
``` 
sudo chmod 666 /dev/ttyACM1  
sudo chmod 666 /dev/ttyACM0
```
