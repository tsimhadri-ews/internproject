import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


#get the most recent train weights .pt
model = YOLO("runs/detect/train7/weights/best.pt")  


st.title("Find Spongebob")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    
    results = model(image)
    
    #x,y,width,height
    for r in results:
        for box in r.boxes:
            coordinates = (box.xyxy).tolist()[0]
            left, top, right, bottom = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
            cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display image.
    st.image(image_rgb, caption='Processed Image', use_column_width=True)
