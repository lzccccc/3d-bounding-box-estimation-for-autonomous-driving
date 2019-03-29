# 3D Bounding Box Estimation for Autonomous Drinving
This project fully implemented paper "*3D Bounding Box Estimation Using Deep Learning and Geometry*" based on previous work by
image-to-3d-bbox(https://github.com/experiencor/image-to-3d-bbox).

## Depandency:
- Python 3.6
- Tensorflow 1.12.0

## Modifications and Improvements:
1. No prior knowledge of the object location is needed. Instead of reducing configuration numbers to 64, the location of 
each object is solved analytically based on local orientation and 2D location.

2. Add soft constraints to improve the stability of 3D bounding box at certain locations.

3. MobileNetV2 backend is used to significantly reduce parameter numbers and make the model 
Fully Convolutional.

4. The orientation loss is changed to the correct form.

5. Bird-eye view visualization is added.



## Results on KITTI raw data: 
MobilenetV2 with ground truth 2D bounding box. 
![347.png](result/0000000347.png)

Video: https://www.youtube.com/watch?v=IIReDnbLQAE


## Train and Evaluate:
First prepare your KITTI dataset in the following format:
```plain
kitti_dateset/
├── 2011_09_26
│   └── 2011_09_26_drive_0084_sync
│           ├── box_3d       <- predicted data
│           ├── calib_02
│           ├── calib_cam_to_cam.txt
│           ├── calib_velo_to_cam.txt
│           ├── image_02
│           ├── label_02
│           └── tracklet_labels.xml
│
└── training
    ├── box_3d    <- predicted data
    ├── calib
    ├── image_2
    └── label_2

```

To train:
1. Specify parameters in `config.py`.
2. run `train.py` to train the model:
```bash
python3 train.py
```

To predict:
1. Change dir in `read_dir.py` to your prediction folder.
2. run `prediction.py` to predict 3D bounding boxes. Change `-d` to your dataset directory, 
`-a` to specify which type of dataset(train/val split or raw), `-w` to specify the training
weights. 

To visualize 3D bounding box:
1. run `visualization3Dbox.py`. Specify `-s` to if save figures or 
view the plot , specify `-p` to your output image folders. 

## Performance:
<table>
  <tr>
    <th colspan="4"></th> 
    <th colspan="3">w/o soft constraint</th>
    <th colspan="3">w/ soft constraint</th> 
  </tr>
  <tr>
    <th>backbone</th>
    <th>parameters / model size</th> 
    <th>inference time(s/img)(cpu/gpu)</th>
    <th>type</th>
    <th>Easy</th>
    <th>Mode</th>
    <th>Hard</th>
    <th>Easy</th>
    <th>Mode</th>
    <th>Hard</th>
  </tr>
  <tr>
    <td rowspan="4">VGG</td>
    <td rowspan="4">40.4 mil. / 323 MB</td> 
    <td rowspan="4">2.041 / 0.081</td>
    <td>AP2D</td>
    <td>100</td>
    <td>100</td>
    <td>100</td>
    <td>100</td>
    <td>100</td>
    <td>100</td>
  </tr> 
  <tr> 
    <td>AOS</td>
    <td>99.98</td>
    <td>99.82</td>
    <td>99.57</td>
    <td>99.98</td>
    <td>99.82</td>
    <td>99.57</td>
  </tr> 
  <tr> 
    <td>APBV</td>
    <td>26.42</td>
    <td>28.15</td>
    <td>27.74</td>
    <td>32.89</td>
    <td>29.40</td>
    <td>33.46</td>
  </tr> 
  <tr> 
    <td>AP3D</td>
    <td>20.53</td>
    <td>22.17</td>
    <td>25.71</td>
    <td>27.04</td>
    <td>27.62</td>
    <td>27.06</td>
  </tr> 
  
  <tr>
    <td rowspan="4">mobileNet v2</td>
    <td rowspan="4">2.2 mil. / 19 MB</td> 
    <td rowspan="4">0.410 / 0.113</td>
    <td>AP2D</td>
    <td>100</td>
    <td>100</td>
    <td>100</td>
    <td>100</td>
    <td>100</td>
    <td>100</td>
  </tr>
  <tr> 
    <td>AOS</td>
    <td>99.78</td>
    <td>99.23</td>
    <td>98.18</td>
    <td>99.78</td>
    <td>99.23</td>
    <td>98.18</td>
  </tr> 
  <tr> 
    <td>APBV</td>
    <td>11.04</td>
    <td>8.99</td>
    <td>10.51</td>
    <td>11.62</td>
    <td>8.90</td>
    <td>10.42</td>
  </tr> 
  <tr> 
    <td>AP3D</td>
    <td>7.98</td>
    <td>7.95</td>
    <td>9.32</td>
    <td>10.42</td>
    <td>7.99</td>
    <td>9.32</td>
  </tr>  
</table>  
Offline Evaluation: 50% for training / 50 % for testing <br>
cpu: core i5 7th <br>
gpu: NVIDIA TITAN X