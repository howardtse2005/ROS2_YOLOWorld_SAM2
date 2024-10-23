import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import torch
import cv2
import os
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
import matplotlib.pyplot as plt

class MultipleObjectsNode(Node):
    def __init__(self):
        super().__init__('multiple_objects_node')
        self.subscription = self.create_subscription(
            Image,
            'input_image_topic', # Change into the image topic name you want to subscribe to
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(Image, 'output_image_topic', 10) # Change into the topic name you want to publish to
        self.bridge = CvBridge()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the SAM model using an absolute path
        sam_checkpoint = os.path.expanduser("~/YOLO-World-SAM_segmentation/sam_vit_h_4b8939.pth") # Here I am using the absolute path to the checkpoint file
        model_type = "vit_h"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=self.device)
        
        # Load YOLO model
        self.yolo_model = YOLO("yolov8s-world.pt")
        
        self.get_logger().info('Node has been initialized')

    def listener_callback(self, msg):
        self.get_logger().info('Received image')
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Run YOLO-world bounding box prediction based on keywords
        classes = ['person', 'bus'] # Change the keywords as you desire
        self.yolo_model.set_classes(classes)
        results = self.yolo_model(cv_image)
        
        # Assign bounding boxes to a list
        all_boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                all_boxes.append([x1, y1, x2, y2])
        
        # Convert the accumulated bounding boxes to a tensor
        input_boxes = torch.tensor(all_boxes, device=self.device)
        transformed_boxes = self.sam.transform.apply_boxes_torch(input_boxes, cv_image.shape[:2])
        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        
        # # Visualize the masks and bounding boxes
        # plt.figure(figsize=(10, 10))
        # plt.imshow(cv_image)
        # for mask in masks:
        #     self.show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        # for box in input_boxes:
        #     self.show_box(box.cpu().numpy(), plt.gca())
        
        # Convert the processed image back to BGR for ROS publishing
        output_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        output_msg = self.bridge.cv2_to_imgmsg(output_image, encoding='bgr8')
        self.publisher.publish(output_msg)

    # def show_mask(self, mask, ax, random_color=False):
    #     if random_color:
    #         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    #     else:
    #         color = np.array([30/255, 144/255, 255/255, 0.6])
    #     h, w = mask.shape[-2:]
    #     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    #     ax.imshow(mask_image)
    
    # def show_box(self, box, ax):
    #     x0, y0 = box[0], box[1]
    #     w, h = box[2] - box[0], box[3] - box[1]
    #     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def main(args=None):
    rclpy.init(args=args)
    node = MultipleObjectsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()