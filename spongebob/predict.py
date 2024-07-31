
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2 

#model = YOLO("yolov8n.yaml")
model = YOLO("/home/ubuntu/objectdetection/runs/detect/train7/weights/best.pt")  
#model = YOLO("yolov8n.yaml").load("yolov8n.pt")


results = model("/home/ubuntu/objectdetection/spongtest4.jpg")
img = cv2.imread("/home/ubuntu/objectdetection/spongtest4.jpg")


for r in results:
    for box in r.boxes:
        coordinates = (box.xyxy).tolist()[0]
        left, top, right, bottom = coordinates[0], coordinates[1], coordinates[2], coordinates[3]

        # Draw the rectangle on the image
        cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)

# Save the image
cv2.imwrite('sponge_test_with_boxes.jpg', img)

print("Image saved with bounding boxes.")