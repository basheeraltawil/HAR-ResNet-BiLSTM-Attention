"""
This script processes a video dataset to extract features for machine learning. 
It uses a pre-trained ResNet-18 model to extract features from sampled frames. 
For each video, it samples frames based on motion intensity, extracts features from the frames, 
and then saves the features and corresponding labels into `.npy` files. 
The script handles both training and validation data, and tracks the time taken for processing each video.
"""
"""
Frame sampling is done based on motion intensity detected using optical flow. 
The script analyzes consecutive frames, computes the motion score between them, 
and selects frames with the highest motion. The frames are sampled with a minimum interval to maintain order, 
and frames are added or interpolated if the number of selected frames is less than the maximum limit.
The goal is to select representative frames with significant motion while maintaining video temporal consistency.
"""
import os
import numpy as np
import pandas as pd
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.preprocessing import LabelEncoder
import time  # Importing time module for timing
import yaml
config_path = r'....../configs/config.yaml'
# ---- Load Configurations from config.yaml ----
with open(config_path, "r") as f:
    config = yaml.safe_load(f)



# Define the file path to your dataset
DEVICE = config['DEVICE']
DATA_DIR= config['DATA_DIR']
TRAIN_ANNOTATION_FILE = config['TRAIN_ANNOTATION_FILE']
VAL_ANNOTATION_FILE = config['VAL_ANNOTATION_FILE']
TEST_ANNOTATION_FILE = config['TEST_ANNOTATION_FILE']
MAX_NUM_FRAMES=int(config['MAX_NUM_FRAMES'])
MOTION_THRESHOLD=float(config['MOTION_THRESHOLD'])
EF_OUT_DIR = config['EF_OUT_DIR']

os.makedirs(EF_OUT_DIR, exist_ok=True)  # Ensure the output directory exists

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
# Load the pre-trained ResNet18 model
resnet = models.resnet18(pretrained=True).to(device)
resnet_feat = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
# Function to sample frames based on motion intensity while maintaining order
def sample_frames(video_path, max_num_frames, motion_threshold, sampling_interval=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}. Skipping...")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval_frames = list(range(0, total_frames, sampling_interval))
    motion_scores = []
    frames = []
    prev_frame = None

    for i in interval_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            motion_score = np.sum(np.abs(flow))
            motion_scores.append((i, motion_score))

        frames.append((i, frame))  # Store frame with its index
        prev_frame = curr_gray

    cap.release()

    motion_scores.sort(key=lambda x: x[1], reverse=True)
    selected_indices = []
    selected_frames = []
    last_index = -sampling_interval

    for idx, score in motion_scores:
        if len(selected_indices) < max_num_frames and idx - last_index >= sampling_interval:
            if score >= motion_threshold:
                selected_indices.append(idx)
                last_index = idx

    if len(selected_indices) < max_num_frames:
        num_needed = max_num_frames - len(selected_indices)
        selected_indices.extend(np.linspace(selected_indices[0] if selected_indices else 0,
                                            selected_indices[-1] if selected_indices else total_frames - 1,
                                            num=num_needed, dtype=int).tolist())
    
    selected_indices = sorted(set(selected_indices))
    selected_indices = selected_indices[:max_num_frames]
    selected_frames = [frame for idx, frame in frames if idx in selected_indices]
    
    while len(selected_frames) < max_num_frames:
        selected_frames.append(selected_frames[-1])

    return selected_frames

# Function to extract features from sampled frames using ResNet
def extract_features(frames):
    features = []
    resnet_feat.eval()

    for frame in frames:
        frame_tensor = transform(frame).to(device)
        with torch.no_grad():
            feature_tensor = resnet_feat(frame_tensor.unsqueeze(0))  # Add batch dimension
        features.append(feature_tensor)

    features_tensor = torch.stack(features).to(device)
    features_tensor = torch.flatten(features_tensor, start_dim=1)
    return features_tensor.cpu().numpy()

# Function to process videos from the annotation file and save combined features/labels
def process_videos(annotation_file, output_features_file, output_labels_file, dataset_name):
    annotations = pd.read_csv(annotation_file)
    le = LabelEncoder()

    all_features = []
    all_labels = []
    total_time_for_dataset = 0.0

    for index, row in annotations.iterrows():
        clip_name = row['clip_name']
        clip_path = row['clip_path']
        label = row['label']
        video_path = os.path.join(DATA_DIR, clip_path.strip('/'))
        print(f"Processing video: {clip_name} | Path: {video_path}")
        start_time = time.time()
        
        sampled_frames = sample_frames(video_path, MAX_NUM_FRAMES, MOTION_THRESHOLD)
        features = extract_features(sampled_frames)

        all_features.append(features)
        all_labels.append(label)
        print(f"Processed video: {clip_name} | Frames used: {len(sampled_frames)}")
        video_time = time.time() - start_time
        total_time_for_dataset += video_time
        print(f"Video: {clip_name} | Time: {video_time:.2f} seconds")

    if all_features:
        features_array = np.stack(all_features, axis=0)
        labels_array = np.array(all_labels)

        numerical_labels = le.fit_transform(labels_array)

        np.save(output_features_file, features_array)
        np.save(output_labels_file, numerical_labels)

        print(f"Total time for {dataset_name} dataset: {total_time_for_dataset:.2f} seconds")
    else:
        print(f"No valid samples processed in {dataset_name}.")

    return total_time_for_dataset

# Tracking total time for all datasets
total_experiment_time = 0.0
# Process each dataset
# train_time = process_videos(TRAIN_ANNOTATION_FILE, 
#                             os.path.join(EF_OUT_DIR, 'lstm_train_features.npy'), 
#                             os.path.join(EF_OUT_DIR, 'lstm_train_labels.npy'), "Training")
# total_experiment_time += train_time

# val_time = process_videos(VAL_ANNOTATION_FILE, 
#                           os.path.join(EF_OUT_DIR, 'lstm_val_features.npy'), 
#                           os.path.join(EF_OUT_DIR, 'lstm_val_labels.npy'), "Validation")
# total_experiment_time += val_time

test_time = process_videos(TEST_ANNOTATION_FILE, 
                           os.path.join(EF_OUT_DIR, 'lstm_test_features.npy'), 
                           os.path.join(EF_OUT_DIR, 'lstm_test_labels.npy'), "Testing")
total_experiment_time += test_time

print(f"Total time for the entire experiment: {total_experiment_time:.2f} seconds")
