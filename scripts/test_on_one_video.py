import torch
import torch.nn as nn
import numpy as np
import yaml
import os
import gc
from lstm_model import EnhancedBiLSTMWithAttention
import cv2
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# Clean GPU memory
torch.cuda.empty_cache()
gc.collect()

# Load configurations
config_path = r'......./configs/config.yaml'
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Configuration parameters
HIDDEN_SIZE = int(config["HIDDEN_SIZE"])
NUM_LAYERS = int(config["NUM_LAYERS"])
NUM_HEADS = int(config["NUM_HEADS"])
CHECKPOINT_TEST_DIR = config['CHECKPOINT_TEST_DIR']
CLASS_NAMES_PATH = config['CLASS_NAMES_PATH']
ONE_VIDEO_TEST_FRAMES=int(config['ONE_VIDEO_TEST_FRAMES'])

# Function to load class names
def load_class_names(file_path):
    class_names = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_names.append(parts[1])
    return class_names

# Function to extract features from a video using ResNet18
def extract_features_from_video(video_path, target_frames=ONE_VIDEO_TEST_FRAMES, target_size=(224, 224)):
    """
    Extract features from a video file using ResNet18 as feature extractor.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load ResNet18 and remove the final classification layer
    resnet18 = models.resnet18(pretrained=True)
    feature_extractor = torch.nn.Sequential(*list(resnet18.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine which frames to sample
    frame_indices = np.linspace(0, total_frames-1, target_frames, dtype=int)
    
    # Initialize transform for preprocessing
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Extract features
    all_features = []
    with torch.no_grad():
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {idx}")
                continue
            
            # Convert BGR to RGB and apply transforms
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame)
            input_tensor = transform(pil_image).unsqueeze(0).to(device)
            
            # Extract features
            features = feature_extractor(input_tensor)
            # Reshape from [1, 512, 1, 1] to [512]
            features = features.squeeze().cpu().numpy()
            all_features.append(features)
    
    cap.release()
    
    # Convert to numpy array and reshape
    all_features = np.array(all_features)
    print(f"Extracted features shape: {all_features.shape}")
    
    # Ensure we have the expected number of frames
    if len(all_features) < target_frames:
        print(f"Warning: Only extracted {len(all_features)} frames out of {target_frames} requested")
        # Pad with zeros if we have fewer frames than expected
        padding = np.zeros((target_frames - len(all_features), all_features.shape[1]))
        all_features = np.vstack([all_features, padding])
    
    # Create tensor of shape [1, sequence_length, feature_dim]
    features_tensor = torch.FloatTensor(all_features).unsqueeze(0)
    print(f"Final features tensor shape: {features_tensor.shape}")
    
    return features_tensor

# Main function to predict class for a single video
def predict_single_video(video_path, model_path, class_names, top_k=3):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Extract features from the video
    print(f"Extracting features from video: {video_path}")
    features = extract_features_from_video(video_path)
    features = features.to(device)
    
    # Define model parameters
    input_size = features.shape[-1]
    num_classes = len(class_names)
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = EnhancedBiLSTMWithAttention(
        input_size=input_size, 
        hidden_size=HIDDEN_SIZE, 
        num_layers=NUM_LAYERS, 
        num_heads=NUM_HEADS, 
        num_classes=num_classes
    ).to(device)
    
    # Load the checkpoint weights
    checkpoint = torch.load(model_path, map_location=device)
    model_dict = model.state_dict()
    
    # Remove keys that don't match
    pretrained_dict = {k: v for k, v in checkpoint.items() 
                      if k in model_dict and model_dict[k].shape == v.shape}
    
    # Update the model with the compatible weights
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        outputs, attention_weights = model(features)
        probabilities = nn.Softmax(dim=1)(outputs)
        
        # Get top-k predictions
        topk_probs, topk_indices = torch.topk(probabilities, top_k)
        
    # Print results
    print("\nTop Three Prediction Results:")
    print("-" * 30)
    for i in range(top_k):
        class_idx = topk_indices[0][i].item()
        prob = topk_probs[0][i].item() * 100
        print(f"Top {i+1}: {class_names[class_idx]} - {prob:.2f}%")
    
    return topk_indices.cpu().numpy(), topk_probs.cpu().numpy(), attention_weights

if __name__ == '__main__':
    # Video path
    video_path = '......dataset/Haircut_01.mp4'
    
    # Model path
    model_path = os.path.join(CHECKPOINT_TEST_DIR)
    
    # Load class names
    class_names = load_class_names(CLASS_NAMES_PATH)
    
    # Predict
    indices, probs, attention = predict_single_video(video_path, model_path, class_names)
    
 