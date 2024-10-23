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

class SingleObjectNode(Node):
    def __init__(self):
        super().__init__('single_object_node')
        self.subscription = self.create_subscription(
            Image,
            'input_image_topic',
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(Image, 'output_image_topic', 10)
        self.bridge = CvBridge()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the SAM model
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
        
        # YOLO detection
        classes = ['bus'] # Change the keywords as you desire
        self.yolo_model.set_classes(classes)
        results = self.yolo_model(cv_image)
        boxes = results.xyxy[0].cpu().numpy()  # Assuming single image and single batch
        
        if len(boxes) > 0:
            box = boxes[0]  # Take the first detected box
            x0, y0, x1, y1 = map(int, box[:4])
            cropped_image = cv_image[y0:y1, x0:x1]
            
            # SAM segmentation
            predictor = SamPredictor(self.sam)
            predictor.set_image(cropped_image)
            masks, _, _ = predictor.predict(box=box[:4])
            
            # Visualize results
            self.show_box(box, cv_image)
            self.show_mask(masks[0], cv_image)
            
            # Convert the processed image back to BGR for ROS publishing
            output_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            output_msg = self.bridge.cv2_to_imgmsg(output_image, encoding='bgr8')
            self.publisher.publish(output_msg)

    def show_mask(self, mask, image, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        image[mask] = (image[mask] * 0.5 + mask_image[mask] * 0.5).astype(np.uint8)
    
    def show_box(self, box, image):
        x0, y0 = int(box[0]), int(box[1])
        x1, y1 = int(box[2]), int(box[3])
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)

def main(args=None):
    rclpy.init(args=args)
    node = SingleObjectNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()