#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from mask_depth_service_interfaces.srv import MaskDepth
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class MaskDepthClient(Node):

    def __init__(self):
        super().__init__('mask_depth_client')
        self.cli = self.create_client(MaskDepth, 'mask_depth')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = MaskDepth.Request()
        self.bridge = CvBridge()

    def send_request(self, rgb_image_path, depth_image_path, x, y):
        # Load the RGB and depth images
        rgb_image = cv2.imread(rgb_image_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

        # Convert OpenCV images to ROS Image messages
        self.req.rgb_image = self.bridge.cv2_to_imgmsg(rgb_image, encoding='rgb8')
        self.req.depth_image = self.bridge.cv2_to_imgmsg(depth_image, encoding='passthrough')

        # Load the point coordinates
        self.req.point = Point(x=x, y=y)

        # Send the request
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    mask_depth_client = MaskDepthClient()

    # Set the paths to the RGB and depth images
    base_dir = os.path.expanduser('~/anygrasp_sdk/grasp_detection/example_data')
    rgb_image_filename = 'color.png'
    depth_image_filename = 'depth.png'
    rgb_image_path = os.path.join(base_dir, rgb_image_filename)
    depth_image_path = os.path.join(base_dir, depth_image_filename)

    # Set the point coordinates
    x = 900.0
    y = 507.0

    response = mask_depth_client.send_request(rgb_image_path, depth_image_path, x, y)
    if response is not None:
        masked_depth_image = mask_depth_client.bridge.imgmsg_to_cv2(response.masked_depth_image, desired_encoding='passthrough')
        cv2.imshow('Masked Depth Image', masked_depth_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        mask_depth_client.get_logger().info('Service call failed')

    mask_depth_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()