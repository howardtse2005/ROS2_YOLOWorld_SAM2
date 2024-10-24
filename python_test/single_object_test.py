import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO

### FUNCTION TO VISUALIZE MASKS, POINTS, AND BOXES ###
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

### LOAD THE MODEL, IMAGE DATA AND PATH ###
# Load the SAM model
sam_checkpoint = os.path.expanduser("~/ROS2_YOLOWorld-SAM/sam_vit_h_4b8939.pth") # Here I am using the absolute path to the checkpoint file
model_type = "vit_h"

# Load YOLO-world model
model = YOLO("yolov8s-world.pt")

# Load the image data path (change this to your desired path)
base_dir = os.path.expanduser('~/ROS2_YOLOWorld-SAM/example_image_dataset') # Change this absolute path to your desired path
color_image_filename = input("Enter the image filename: ")
color_image_path = os.path.join(base_dir, color_image_filename)

# Load the image
image = cv2.imread(color_image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Predict the masks by SAM model
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
predictor.set_image(image)

# Run YOLO-world bounding box prediction based on a keyword
keyword = input("Enter the keyword: ")
keyword_list = [keyword]
model.set_classes(keyword_list)

results = model.predict(color_image_path)
result = results[0]

# Check if any bounding boxes are detected
if len(result.boxes) == 0:
    print("No objects detected!")
else:
    ### PREDICT MASKS BASED ON A BOUNDING BOX ###
    input_box = result.boxes[0].xyxy[0].cpu().numpy() 
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )

    # Visualize the masks and bounding box
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    show_box(input_box, plt.gca())
    plt.axis('off')
    plt.show()
