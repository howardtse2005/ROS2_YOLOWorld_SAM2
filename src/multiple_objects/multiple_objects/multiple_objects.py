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

class MultipleObjectsNode(Node):
    def __init__(self):
        super().__init__('multiple_objects_node')
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw', # Change into the image topic name you want to subscribe to
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(Image, 'output_image_topic', 10) # Change into the topic name you want to publish to
        self.bridge = CvBridge()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the SAM model using an absolute path
        sam2_checkpoint = os.path.expanduser("~/YOLOWorld_SAM2_ws/checkpoints/sam2.1_hiera_tiny.pt")
        model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        self.sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        
        # Load YOLO model
        self.yolo_model = YOLO("yolov8s-world.pt")
        
        self.get_logger().info('Node has been initialized')
        
        # Ask user to input keywords
        keywords_input = input("Enter keywords separated by commas: ")
        self.classes = [cls.strip() for cls in keywords_input.split(",")]

    def listener_callback(self, msg):
        self.get_logger().info('Received image')
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Set the image for the predictor
        self.predictor.set_image(cv_image)
        
        # Run YOLO-world bounding box prediction based on keywords
        self.yolo_model.set_classes(self.classes)
        results = self.yolo_model(cv_image)
        
        # Assign bounding boxes to a list
        all_boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Convert to list and extract coordinates
                all_boxes.append([x1, y1, x2, y2])

        # Check if any bounding boxes are detected
        if len(all_boxes) == 0:
            self.get_logger().warn('No objects detected!')
            # Publish original image
            output_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            output_msg = self.bridge.cv2_to_imgmsg(output_image, encoding='bgr8')
            self.publisher.publish(output_msg)
            return
        
        # Predict masks
        input_boxes = np.array(all_boxes)
        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
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

        for box in input_boxes:
            x0, y0, x1, y1 = box
            cv2.rectangle(cv_image, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)

        # Convert the processed image back to BGR for publishing
        output_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        output_msg = self.bridge.cv2_to_imgmsg(output_image, encoding='bgr8')
        
        # Publish the output image
        self.publisher.publish(output_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MultipleObjectsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()