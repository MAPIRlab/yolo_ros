from ultralytics import YOLO
import random
import cv2
import numpy as np


## Minimal example, just to check that yolo is propely installed in your system and added to the PYTHONPATH
# based on https://medium.com/@Mert.A/how-to-segment-objects-with-yolov9-71a3439c8b10

def __main__():
    
    model = YOLO("yolov9e-seg.pt")

    img = cv2.imread("yolo_ros/resources/Untitled.png")

    # choose which classes you want
    yolo_classes = list(model.names.values())
    classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]

    conf = 0.2
    
    # run inference
    results = model.predict(img, conf=conf)
    
    colors = [random.choices(range(256), k=3) for _ in classes_ids]
    print(results)
    for result in results:
        for mask, box in zip(result.masks.xy, result.boxes):
            points = np.int32([mask])
            # cv2.polylines(img, points, True, (255, 0, 0), 1)
            color_number = classes_ids.index(int(box.cls[0]))
            cv2.fillPoly(img, points, colors[color_number])
    
    cv2.imwrite("yolo.png", img)

__main__()