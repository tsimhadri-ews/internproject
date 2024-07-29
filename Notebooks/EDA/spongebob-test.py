import cv2
import os

# Initialize the bounding box coordinates
bbox = []
drawing = False

def draw_rectangle(event, x, y, flags, param):
    global bbox, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        bbox = [(x, y)]
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img.copy()
            cv2.rectangle(img_copy, bbox[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("Image", img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        bbox.append((x, y))
        drawing = False
        cv2.rectangle(img, bbox[0], bbox[1], (0, 255, 0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

# Read the image
image_path = r'~/internproject/Data/spongebob-test.jpg' 
img = cv2.imread(os.path.expanduser(image_path))
cv2.imshow("Image", img)
cv2.setMouseCallback("Image", draw_rectangle)

print("Draw a bounding box on the image and press 'q' to quit.")
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()

label_path = os.path.splitext(image_path)[0] + ".txt"

if len(bbox) == 2:
    x1, y1 = bbox[0]
    x2, y2 = bbox[1]
    img_height, img_width, _ = img.shape
    
    # Normalize coordinates
    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height
    width = abs(x2 - x1) / img_width
    height = abs(y2 - y1) / img_height
    
    # Format coordinates
    label_data = f"0 {x_center} {y_center} {width} {height}"
else:
    # No bounding box was drawn
    label_data = "1 0 0 0 0"

# Write to text file
with open(os.path.expanduser(label_path), "w") as file:
    file.write(label_data)
    
print(f"Bounding box coordinates saved to {label_path}: {label_data}")
