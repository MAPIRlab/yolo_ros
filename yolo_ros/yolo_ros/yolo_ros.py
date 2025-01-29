#!/usr/bin/env python3
import time
import os
import rclpy
import rclpy.node
import sensor_msgs.msg
from ament_index_python.packages import get_package_share_directory

from ultralytics import YOLO
from segmentation_msgs.srv import SegmentImage
from segmentation_msgs.msg import SemanticInstance2D
from vision_msgs.msg import Detection2D, BoundingBox2D, ObjectHypothesisWithPose

import torch
import numpy as np
from cv_bridge import CvBridge
import cv2
'''
Params:
interest_classes: ([int]) list of COCO class indices that we want to output. Defaults to all
visualization: (bool) whether to publish the segmented image on a topic
visualization_topic: (str) pretty self-explanatory
model_file: (str) path to the model yaml relative to yolo2/configs. You do not need to download the weights separately, yolo handles that itself
'''

class Yolo_ros (rclpy.node.Node):

    def __init__(self):
        super().__init__('Yolo_ros')
        
        # list of COCO class indices that we want to output. Defaults to all
        self.interest_classes = self.load_param("interest_classes", [*range(80)])  
        
        self.publish_visualization = self.load_param("publish_visualization", True) 
        visualization_topic = self.load_param("visualization_topic", "/yolo/segmentedImage")
        self.visualization_pub = self.create_publisher(sensor_msgs.msg.Image, visualization_topic, 1)

        self.cv_bridge = CvBridge()
        
        MODEL_FILE=self.load_param("model_file", "yolov9e-seg.pt")
        self.set_up_yolo(MODEL_FILE)

        self.segment_image_srv =  self.create_service(SegmentImage, "/yolo/segment", self.segment_image)
        self._logger.info("Done setting up!")
        self._logger.info(f"Advertising service: {self.segment_image_srv.srv_name}")


    def set_up_yolo(self, MODEL_FILE):
        self._logger.info(f"Yolo model file: {MODEL_FILE}")
        
        # prepare the 'models' folder to download the selected model
        pkg_dir = get_package_share_directory("yolo_ros")
        models_dir = os.path.join(pkg_dir, "models")
        if not os.path.exists(models_dir):
            os.mkdir(models_dir)
        
        # Create the predictor
        self.yolo = YOLO(os.path.join(models_dir, MODEL_FILE))

        #choose the classes of interest
        #TODO check if we can filter the predictor output based on this 
        self._class_names = list(self.yolo.names.values())
        interest_class_names = [self._class_names[i] for i in self.interest_classes]
        
        self._logger.info(f"Classes of interest: {interest_class_names}")



    def segment_image(self, request, response):
        self._logger.info("Received image")
        try:
            numpy_image = self.cv_bridge.imgmsg_to_cv2(request.image)
        except Exception as e:
            self._logger.warn(f"Exception when trying to process image!\n Exception: {e}")
            return response
        
        conf = 0.2
        
        # run inference
        results_list = self.yolo.predict(numpy_image, conf=conf)

        for result in results_list:       
            if result.masks is  None or result.boxes is None: 
                continue

            for mask, box in zip(result.masks.xy, result.boxes):
                # Get the mask pixel coordinates in the right format to be used as indices
                mask = np.int32([mask])[0]
                col0 = mask[:, 0]
                col1 = mask[:, 1]
                mask[:, 0] = col1
                mask[:, 1] = col0
                
                semantic_instance = SemanticInstance2D()
                msg_mask = np.zeros(result.masks.orig_shape, dtype="uint8")
                msg_mask[mask[:, :]] = 255

                semantic_instance.mask = self.cv_bridge.cv2_to_imgmsg(msg_mask)
                class_id = int(box.cls[0])
                semantic_instance.detection = self.set_singleclass_detection(result.names[class_id], box.conf[0], box)

                response.instances.append(semantic_instance)

        if self.publish_visualization:
            viz_img = numpy_image
            for result in results_list:      
                viz_img = result.plot()

            image_msg_a = self.cv_bridge.cv2_to_imgmsg(viz_img)
            self.visualization_pub.publish(image_msg_a)

        return response



    def set_singleclass_detection(self, class_name, score, bbox):

        detection = Detection2D()
        detection.bbox = BoundingBox2D()
        x,y,w,h = bbox.xywh[0]
        detection.bbox.center.position.x = float(x) + float(w) * 0.5  
        detection.bbox.center.position.y = float(y) + float(h) * 0.5
        detection.bbox.size_x = float(w)
        detection.bbox.size_y = float(h)
        detection.results = []
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = class_name
        hypothesis.hypothesis.score = float(score)

        detection.results.append(hypothesis)

        return detection


    def load_param(self, param, default=None):
        param_value = self.declare_parameter(param, default).value
        # self.get_logger().info("{}: {}".format(param, param_value))
        return param_value


def main(args=None):
    rclpy.init(args=args)
    node = Yolo_ros()

    rclpy.spin(node)

    node.destroy_node()


if __name__ == '__main__':
    main()
