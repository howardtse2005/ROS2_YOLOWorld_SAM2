# Object Detection and Segmentation
This repository combines YOLO-World and SAM 2 (Segment Anything 2) to perform object segmentation based on keywords (zero-shot) and implement it in ROS2 Humble. YOLO-world will detect object and create bounding boxes, then SAM will create segmentation masks based on the bounding boxes.

Before using this repository, make sure you install the dependencies needed for YOLO-World and SAM 2. You can refer to the following github repos: 

YOLO-World: https://github.com/AILab-CVC/YOLO-World.git

SAM 2: https://github.com/facebookresearch/sam2.git

To use this repository, follow the following steps:
1. Install Ultralytics for YOLO World.
```
pip install ultralytics
```
2. Git clone this repository (I suggest to clone to your root directory (~) or you need to update the absolute path in the codes).
```
git clone https://github.com/howardtse2005/ROS2_YOLOWorld_SAM2.git
```
3. Install SAM 2 inside the src/ directory
```
cd src/

git clone https://github.com/facebookresearch/sam2.git && cd sam2

pip install -e .

pip install -e ".[notebooks]"
```
4. Install SAM model checkpoints:
```
cd ../..

cd checkpoints && \
./download_ckpts.sh && \
cd ..
```
5. To run the ROS2 codes
```
colcon build

source install/setup.bash

ros2 run single_object single_object_node # For single object detection. Input 1 keyword and it will detect the object with the highest score

ros2 run multiple_objects multiple_objects_node # For multiple objects detection. Input several keywords seperated by commas.
```
Make sure you change the input topic name to the topic you want to subscribe to.
