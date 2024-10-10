#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from mask_depth_service_interfaces.srv import MaskDepth
import numpy as np
import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor
import os

class MaskDepthService(Node):

    def __init__(self):
        super().__init__('mask_depth_service')
        self.srv = self.create_service(MaskDepth, 'mask_depth', self.mask_depth_callback)
        # Initialize CV Bridge and model checkpoint
        self.bridge = CvBridge()
        self.sam_checkpoint = "sam_vit_h_4b8939.pth"
        self.model_type = "vit_h"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = self.load_model()
        # Initialize the paths to the masked depth image
        self.base_dir = os.path.expanduser('~/anygrasp_sdk/grasp_detection/example_data')
        self.depth_image_modified_filename = 'depth_masked.png'
        self.depth_image_modified_path = os.path.join(self.base_dir, self.depth_image_modified_filename)

    def load_model(self):
        # Load the model predictor
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)
        predictor = SamPredictor(sam)
        return predictor

    def mask_depth_callback(self, request, response):
        # Convert ROS Image messages to OpenCV images
        rgb_image = self.bridge.imgmsg_to_cv2(request.rgb_image, desired_encoding='rgb8')
        depth_image = self.bridge.imgmsg_to_cv2(request.depth_image, desired_encoding='passthrough')

        ### MASKING THE RGB IMAGE ###
        # Set the image for the predictor
        self.predictor.set_image(rgb_image)

        # Predict masks based on the input point and its label
        input_point = np.array([[request.point.x, request.point.y]])
        input_label = np.array([1])

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        ### MASKING THE DEPTH IMAGE ###
        masked_depth_image = depth_image.copy()
        
        # Convert the first mask to a boolean array
        boolean_mask = masks[0].astype(bool)

        # Set depth pixels outside the masked area to 100000
        masked_depth_image[~boolean_mask] = 100000

        # Save the masked depth image (optional)
        cv2.imwrite(self.depth_image_modified_path, masked_depth_image)

        # Convert the masked depth image to a ROS Image message
        response.masked_depth_image = self.bridge.cv2_to_imgmsg(masked_depth_image, encoding="passthrough")
        return response

def main(args=None):
    rclpy.init(args=args)
    node = MaskDepthService()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()