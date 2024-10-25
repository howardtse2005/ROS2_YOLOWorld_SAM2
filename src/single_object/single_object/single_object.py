import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import torch
import cv2
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ultralytics import YOLO
import matplotlib.pyplot as plt

class SingleObjectNode(Node):
    def __init__(self):
        super().__init__('single_object_node')
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw', # Change into the image topic name you want to subscribe to
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(Image, 'output_image_topic', 10) # Change into the topic name you want to publish to
        self.bridge = CvBridge()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the SAM model
        sam2_checkpoint = os.path.expanduser("~/YOLOWorld_SAM2_ws/checkpoints/sam2.1_hiera_tiny.pt")
        model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        self.sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        
        # Load YOLO model
        self.yolo_model = YOLO("yolov8s-world.pt")

        self.get_logger().info('Node has been initialized')

        # Ask user to input keyword
        keyword = input("Enter the keyword: ")
        keyword_list = [keyword]
        self.classes = keyword_list

    def listener_callback(self, msg):
        self.get_logger().info('Received image')
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Set the image for the predictor
        self.predictor.set_image(cv_image)
        
        # Run YOLO-world bounding box prediction based on keyword
        self.yolo_model.set_classes(self.classes)
        results = self.yolo_model(cv_image)
        result = results[0]

        # Check if any bounding boxes are detected
        if len(result.boxes) == 0:
            self.get_logger().warn('No objects detected!')
            # Publish original image
            output_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            output_msg = self.bridge.cv2_to_imgmsg(output_image, encoding='bgr8')
            self.publisher.publish(output_msg)
            return
        
        # Predict masks
        input_box = result.boxes[0].xyxy[0].cpu().numpy()
        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=False,
        )
        # Draw masks and bounding boxes on the image
        for mask in masks:
            color = np.array([30/255, 144/255, 255/255, 0.6])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            mask_image = (mask_image * 255).astype(np.uint8)
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_RGBA2RGB)
            cv_image = cv2.addWeighted(cv_image, 1.0, mask_image, 0.6, 0)

            x0, y0, x1, y1 = input_box
            cv2.rectangle(cv_image, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)

        # Convert the processed image back to BGR for publishing
        output_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        output_msg = self.bridge.cv2_to_imgmsg(output_image, encoding='bgr8')

        # Publish the output image
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