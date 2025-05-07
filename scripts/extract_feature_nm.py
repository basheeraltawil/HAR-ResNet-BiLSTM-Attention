"""
This script extracts features from video clips using a pre-trained ResNet-18 model.
It processes videos based on given annotations and samples frames at regular intervals.
Features are extracted from the sampled frames and saved along with corresponding labels.
The output is stored as `.npy` files for training, validation, or testing. 
Log files track processing time and sampled frames for each dataset.

Frame sampling selects evenly spaced frames across the video. If there are fewer frames than required,
the script duplicates frames to meet the `num_frames` requirement. 
Frames are resized, normalized, and converted into tensors before feature extraction.
"""
import os
import pandas as pd
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.preprocessing import LabelEncoder
import time
import os
import yaml
config_path = r'/media/basheer/OVGU/Projects/HAR_Project/cluster/github_repo/configs/config.yaml'
# ---- Load Configurations from config.yaml ----
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

DEVICE = config['DEVICE']
DATA_DIR= config['DATA_DIR']
TRAIN_ANNOTATION_FILE = config['TRAIN_ANNOTATION_FILE']
VAL_ANNOTATION_FILE = config['VAL_ANNOTATION_FILE']
TEST_ANNOTATION_FILE = config['TEST_ANNOTATION_FILE']
MAX_NUM_FRAMES=int(config['MAX_NUM_FRAMES'])
MOTION_THRESHOLD=float(config['MOTION_THRESHOLD'])
EF_OUT_DIR = config['EF_OUT_DIR']
TRAIN_LOG_FILE = config['TRAIN_LOG_FILE']
VAL_LOG_FILE = config['VAL_LOG_FILE']
TEST_LOG_FILE = config['TEST_LOG_FILE']

os.makedirs(EF_OUT_DIR, exist_ok=True)  # Ensure the output directory exists


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the pre-trained ResNet18 model
resnet = models.resnet18(pretrained=True).to(DEVICE)
resnet_feat = torch.nn.Sequential(*list(resnet.children())[:-1]).to(DEVICE)

# Function to sample frames based on motion intensity while maintaining order
def extract_features(annotation_file, output_features, output_labels, log_file):
    samples = []
    resnet_feat.eval()
    total_time = 0  # To accumulate time for all videos

    # Load the annotations
    annotations = pd.read_csv(annotation_file)

    with open(log_file, 'w') as log:  # Open the log file for writing
        for _, row in annotations.iterrows():
            clip_name = row['clip_name']
            clip_path = row['clip_path']
            label = row['label']

            video_path = os.path.join(DATA_DIR, clip_path.strip('/'))
            
            print(f"Processing video: {clip_name} | Path: {video_path}")

            start_time = time.time()  # Start timing the video processing

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video {video_path}. Skipping...")
                continue

            # Get total number of frames in the video
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate interval for evenly sampling frames
            frame_indices = np.linspace(0, total_frames - 1, min(total_frames, MAX_NUM_FRAMES), dtype=int)
            frames = []

            # Capture and store frames
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # Set the position of the next frame
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = transform(frame).to(DEVICE)
                    frames.append(frame)

            # Repeat frames if not enough to reach num_frames
            while len(frames) < MAX_NUM_FRAMES:
                print("filling the frames")
                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = transform(frame).to(DEVICE)
                        frames.append(frame)
                    if len(frames) == MAX_NUM_FRAMES:
                        break  # Stop once we have enough frames

            cap.release()

            if len(frames) == MAX_NUM_FRAMES:
                frames_tensor = torch.stack(frames).to(DEVICE)
                with torch.no_grad():
                    features_tensor = resnet_feat(frames_tensor)
                features_tensor = torch.flatten(features_tensor, start_dim=1)
                features = features_tensor.cpu().numpy()
                samples.append((features, label))

                # Calculate processing time
                end_time = time.time()
                time_needed = end_time - start_time
                total_time += time_needed

                # Log video information
                log_message = f"{clip_name}, {total_frames}, {time_needed:.2f} seconds, sampled frames indices: {list(frame_indices)}"
                log.write(log_message + '\n')

                # Print to terminal
                print(log_message)
            else:
                print(f"Warning: Not enough frames sampled for video {clip_name}.")

    # Convert samples to NumPy arrays
    if samples:
        features_array = np.array([s[0] for s in samples])
        labels_array = np.array([s[1] for s in samples])
        
        le = LabelEncoder()
        numerical_labels = le.fit_transform(labels_array)

        np.save(output_features, features_array)
        np.save(output_labels, numerical_labels)

        print("Features shape:", features_array.shape)
        print("Labels shape:", numerical_labels.shape)
    else:
        print("Warning: No samples processed from the annotation file.")

    print(f"Total time for all videos in {annotation_file}: {total_time:.2f} seconds")




# Process training data
extract_features(TRAIN_ANNOTATION_FILE, 
                 os.path.join(EF_OUT_DIR, 'train_features.npy'), 
                 os.path.join(EF_OUT_DIR, 'train_labels.npy'),
                 TRAIN_LOG_FILE)

# Process validation data
extract_features(VAL_ANNOTATION_FILE, 
                 os.path.join(EF_OUT_DIR, 'val_features.npy'), 
                 os.path.join(EF_OUT_DIR, 'val_labels.npy'),
                 VAL_LOG_FILE)

# Process testing data
extract_features(TEST_ANNOTATION_FILE, 
                 os.path.join(EF_OUT_DIR, 'test_features.npy'), 
                 os.path.join(EF_OUT_DIR, 'test_labels.npy'),
                 TEST_LOG_FILE)
