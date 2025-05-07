#!/usr/bin/env python3         
import time
import os
import gc
from collections import Counter
## for handling torch vision model and open cv, make cv2 before importing torch and torch vision 
import cv2
cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
import torch
import torch.quantization
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F  # For softmax
import torchvision.models as models
from torchvision import transforms
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score, balanced_accuracy_score
from lstm_model import EnhancedBiLSTMWithAttention
# Free up GPU memory if needed
torch.cuda.empty_cache()
gc.collect()
# Set OpenCV backend options
cv2.setUseOptimized(True)
cv2.setNumThreads(4)  # Use 4 threads for better parallelization while avoiding overhead
import yaml
# Load configurations
config_path = r'....../configs/config.yaml'
# ---- Load Configurations from config.yaml ----
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
# Load the ResNet-18 model for feature extraction
CHECKPOINT_TEST_DIR=config['CHECKPOINT_TEST_DIR'] 
CLASS_NAMES_PATH=config['CLASS_NAMES_PATH']
DEVICE = config['DEVICE']
model_path = os.path.join(CHECKPOINT_TEST_DIR)

device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')

#parameters
HIDDEN_SIZE=int(config["HIDDEN_SIZE"])
NUM_LAYERS=int(config["NUM_LAYERS"])
NUM_HEADS=int(config["NUM_HEADS"])
# Utility Functions
def load_class_names(file_path):
    class_names = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                class_names.append(parts[1])
        return class_names
    except Exception as e:
        print(f"Error loading class names: {e}")
        return []

# Load the ResNet-18 model for feature extraction
def load_resnet18(device):
    
    try:
        # Use a smaller model for better inference speed
        resnet18 = models.resnet18(pretrained=True)
        # Remove the final classification layer
        feature_extractor = torch.nn.Sequential(*list(resnet18.children())[:-1])
        feature_extractor = feature_extractor.to(device)
        feature_extractor.eval()
        
        # Apply quantization for better inference speed (if not using CUDA)
        if device.type != 'cuda':
            try:
                
                feature_extractor_quantized = torch.quantization.quantize_dynamic(
                    feature_extractor, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
                )
                return feature_extractor_quantized
            except Exception as e:
                print(f"Quantization not available, using standard model: {e}")
        
        return feature_extractor
    except Exception as e:
        print(f"Error loading ResNet-18: {e}")
        raise

# Load the custom model
def load_model(model_path, input_size, hidden_size, num_layers, num_heads, num_classes, device):
    try:
        model = EnhancedBiLSTMWithAttention(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads, num_classes=num_classes).to(device)
        
        # Load checkpoint
        if device.type == 'cuda':
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location=device)
            
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        # Set to evaluation mode
        model.eval()  
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Preprocess frame transformation - create once and reuse
preprocess_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
])

# Frame preprocessing with caching to avoid repetitive transformations
frame_cache = {}
def preprocess_frame(frame, frame_id=None):
    # Use frame_id for caching if provided
    if frame_id is not None and frame_id in frame_cache:
        return frame_cache[frame_id]
        
    # Process the frame
    tensor_frame = preprocess_transform(frame)
    tensor_frame = tensor_frame.unsqueeze(0)  # Add batch dimension
    
    if frame_id is not None:
        frame_cache[frame_id] = tensor_frame
        if len(frame_cache) > 30:  # Arbitrary limit based on memory constraints
            oldest_key = next(iter(frame_cache))
            del frame_cache[oldest_key]
            
    return tensor_frame

# Main function to test the model using the PC camera
def test_with_camera(model_path, class_names, input_size, hidden_size, num_layers, num_heads, num_classes, device, num_frames=16):
    # Load ResNet-18 for feature extraction
    print("Loading ResNet-18 model...")
    start_time = time.time()
    resnet18 = load_resnet18(device)
    print(f"ResNet-18 loaded in {time.time() - start_time:.2f} seconds")

    # Load the custom model
    print("Loading custom model...")
    start_time = time.time()
    model = load_model(model_path, input_size, hidden_size, num_layers, num_heads, num_classes, device)
    print(f"Custom model loaded in {time.time() - start_time:.2f} seconds")

    if model is None:
        print("Failed to load the model. Exiting.")
        return

    print("Initializing camera...")
    cap = cv2.VideoCapture(0)  # 0 is the default camera
    
    # Optimize camera settings for faster processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution for faster processing
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    else:
        print("Camera opened successfully.")
    
    # Initialize a ring buffer for frames (more efficient than list for fixed size)
    frame_buffer = [None] * num_frames
    buffer_index = 0
    buffer_filled = False
    
 
    
    # Cache for feature extraction
    frame_features_cache = {}
    
    # For skipping frames to increase speed
    frame_count = 0
    frame_skip = 1  # Process every nth frame
    
    # Warmup the models
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_tensor = preprocess_frame(dummy_frame)
    dummy_tensor = dummy_tensor.to(device)
    with torch.no_grad():
        _ = resnet18(dummy_tensor)
    
    print("Starting camera feed loop...")
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        frame_count += 1
        
        # Skip frames for better performance
        if frame_count % frame_skip != 0:
            # Still display the frame
            cv2.imshow('Camera Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
            
       
        
        # Extract features and update buffer
        start_process = time.time()
        
        # Use frame hash for caching
        frame_hash = hash(frame.tobytes()[:1000])  # Use part of the frame for hashing
        
        # Check if features already computed
        if frame_hash in frame_features_cache:
            features = frame_features_cache[frame_hash]
        else:
            # Preprocess the frame for ResNet-18
            input_frame = preprocess_frame(frame)
            input_frame = input_frame.to(device)

            # Extract features using ResNet-18
            with torch.no_grad():
                features = resnet18(input_frame)
                features = features.view(features.size(0), -1)  # Flatten the features
                
            # Cache the features
            frame_features_cache[frame_hash] = features
            
            # Keep cache size manageable
            if len(frame_features_cache) > 50:  # Arbitrary limit
                oldest_key = next(iter(frame_features_cache))
                del frame_features_cache[oldest_key]

        # Update the frame buffer (ring buffer)
        frame_buffer[buffer_index] = features
        buffer_index = (buffer_index + 1) % num_frames
        
        # Check if buffer is filled
        if buffer_index == 0:
            buffer_filled = True
            
        # If the buffer has enough frames, make a prediction
        if buffer_filled:
            # Convert the buffer to a tensor, respecting the temporal order
            ordered_buffer = frame_buffer[buffer_index:] + frame_buffer[:buffer_index]
            # Filter out None values (in case buffer is not completely filled)
            ordered_buffer = [b for b in ordered_buffer if b is not None]
            if len(ordered_buffer) == num_frames:
                sequence = torch.cat(ordered_buffer, dim=0).view(1, num_frames, -1)  # Shape: (batch_size, num_frames, feature_size)
            else:
                # Skip prediction if buffer is not completely filled
                continue

            # Pass the sequence to the custom model
            with torch.no_grad():
                outputs, _ = model(sequence)
                probabilities = F.softmax(outputs, dim=1)  # Convert logits to probabilities
                confidence, predicted = torch.max(probabilities, 1)
                predicted_class = predicted.item()
                confidence_score = confidence.item()

            # Get the class name
            class_name = class_names[predicted_class] if predicted_class < len(class_names) else "Unknown"

            # Display the class name and confidence score on the frame
            cv2.putText(frame, f"Class: {class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence_score:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
     
        process_time = time.time() - start_process
        cv2.putText(frame, f"Process in: {process_time*1000:.1f}ms", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Camera Feed', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()
    
    # Clear caches
    frame_cache.clear()
    frame_features_cache.clear()


# Main
if __name__ == '__main__':
    # Load class names
    print("Loading class names...")
    class_names = load_class_names(r'/media/basheer/OVGU/Datasets/ucf101/classInd.txt')
    print(f"Loaded {len(class_names)} classes")

    # Define model parameters
    input_size = 512  # Example input size for the custom model (adjust based on ResNet-18 output)
    hidden_size = HIDDEN_SIZE
    num_layers = NUM_LAYERS
    num_heads = NUM_HEADS
    num_classes = len(class_names)
    # Number of frames to accumulate before making a decision
    num_frames = 7  # User can change this parameter
    print(f"Using device: {device}")

    # Test the model using the PC camera
    test_with_camera(model_path, class_names, input_size, hidden_size, num_layers, num_heads, num_classes, device, num_frames)