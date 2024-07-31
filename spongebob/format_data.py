import os
import zipfile
import random
import shutil
from sklearn.model_selection import train_test_split

# Define the paths
data_folder = 'Data'
test_zip = os.path.join(data_folder, 'test.zip')
train_zip = os.path.join(data_folder, 'train.zip')
unzip_test_folder = os.path.join(data_folder, 'test')
unzip_train_folder = os.path.join(data_folder, 'train')

# Unzip the files
with zipfile.ZipFile(test_zip, 'r') as zip_ref:
    zip_ref.extractall(unzip_test_folder)

with zipfile.ZipFile(train_zip, 'r') as zip_ref:
    zip_ref.extractall(unzip_train_folder)

# Combine the unzipped folders into a single data folder
combined_data_folder = os.path.join(data_folder, 'combined')
os.makedirs(combined_data_folder, exist_ok=True)

for folder in [unzip_test_folder, unzip_train_folder]:
    for filename in os.listdir(folder):
        shutil.move(os.path.join(folder, filename), combined_data_folder)

# Get a list of image and text file pairs
files = os.listdir(combined_data_folder)
image_files = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]
txt_files = [f for f in files if f.endswith('.txt')]

# Ensure that each image has a corresponding text file
data_pairs = []
for image in image_files:
    txt_file = os.path.splitext(image)[0] + '.txt'
    if txt_file in txt_files:
        data_pairs.append((image, txt_file))

# Split the data into training and testing sets
train_pairs, test_pairs = train_test_split(data_pairs, test_size=0.2, random_state=42)

# Create directories for training and testing data
train_folder = os.path.join(data_folder, 'train_split')
test_folder = os.path.join(data_folder, 'test_split')
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Move the files to the respective folders
for image, txt in train_pairs:
    shutil.move(os.path.join(combined_data_folder, image), train_folder)
    shutil.move(os.path.join(combined_data_folder, txt), train_folder)

for image, txt in test_pairs:
    shutil.move(os.path.join(combined_data_folder, image), test_folder)
    shutil.move(os.path.join(combined_data_folder, txt), test_folder)

# Cleanup: remove the unzipped and combined folders
shutil.rmtree(unzip_test_folder)
shutil.rmtree(unzip_train_folder)
shutil.rmtree(combined_data_folder)

print("Data splitting complete.")
