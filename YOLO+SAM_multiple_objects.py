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
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

# Load YOLO-world model
model = YOLO("yolov8s-world.pt")

base_dir = os.path.expanduser('~/DetectionSegmentationTesting/example_image_dataset')
color_image_filename = input("Enter the image filename: ")
color_image_path = os.path.join(base_dir, color_image_filename)

# Load the RGB image
image = cv2.imread(color_image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
predictor.set_image(image)

# Run YOLO-world bounding box prediction based on set classes
keywords_input = input("Enter classes separated by commas: ")
classes = [cls.strip() for cls in keywords_input.split(",")]
model.set_classes(classes)
results = model.predict(color_image_path)

### PREDICT MASKS BASED ON A BOUNDING BOX ###
# Initialize an empty list to accumulate bounding boxes
all_boxes = []

# Iterate over all results
for result in results:
    # Extract bounding boxes from the result
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # Convert to list and extract coordinates
        all_boxes.append([x1, y1, x2, y2])

# Convert the accumulated bounding boxes to a tensor
input_boxes = torch.tensor(all_boxes, device=predictor.device)
transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
masks, _, _ = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False,
)

# Visualize the masks and bounding box
plt.figure(figsize=(10, 10))
plt.imshow(image)
for mask in masks:
    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
for box in input_boxes:
    show_box(box.cpu().numpy(), plt.gca())
plt.axis('off')
plt.show()