import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

### FUNCTION TO VISUALIZE MASKS, POINTS, AND BOXES ###
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

### LOAD THE MODEL, IMAGE DATA AND PATH ###
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

base_dir = os.path.expanduser('~/DetectionSegmentationTesting/image_dataset')
color_image_filename = 'example_color.jpeg'
depth_image_filename = 'example_depth.jpeg'
depth_image_modified_filename = 'example_depth_modified.jpeg'

color_image_path = os.path.join(base_dir, color_image_filename)
depth_image_path = os.path.join(base_dir, depth_image_filename)
depth_image_modified_path = os.path.join(base_dir, depth_image_modified_filename)

# Load the RGB image
image = cv2.imread(color_image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load the depth image
depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
predictor.set_image(image)

### PREDICT MASKS BASED ON A POINT AND ITS LABEL ###
# For testing purpose, we manually input the point and its label
input_point = np.array([[900, 507]])
input_label = np.array([1])

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
)

# Visualize the masks, points, and labels
for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()

### FILTER OUT DEPTH PIXELS OUTSIDE MASKS ###
# Convert the first mask to a boolean array
boolean_mask = masks[0].astype(bool)

# Set depth pixels outside the masked area to infinity
depth_image[~boolean_mask] = 10000

# Save the modified depth image
cv2.imwrite(depth_image_modified_path, depth_image)

# Visualize the modified depth image
plt.figure(figsize=(10,10))
plt.imshow(depth_image, cmap='gray')
plt.title("Depth Image with Masked Area Set to Infinity", fontsize=18)
plt.axis('off')
plt.show()