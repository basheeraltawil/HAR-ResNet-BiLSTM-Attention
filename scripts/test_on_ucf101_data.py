
    
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score,balanced_accuracy_score
import seaborn as sns
import os
import gc
from lstm_model import EnhancedBiLSTMWithAttention
from collections import Counter
import yaml
torch.cuda.empty_cache()
gc.collect()

config_path = r'....../configs/config.yaml'
# ---- Load Configurations from config.yaml ----
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
#directories
TEST_FEATURES_PATH=config['TEST_FEATURES_PATH']
TEST_LABELS_PATH=config['TEST_LABELS_PATH'] 
CHECKPOINT_TEST_DIR=config['CHECKPOINT_TEST_DIR'] 
CLASS_NAMES_PATH=config['CLASS_NAMES_PATH']
# parameters
HIDDEN_SIZE=int(config["HIDDEN_SIZE"])
NUM_LAYERS=int(config["NUM_LAYERS"])
NUM_HEADS=int(config["NUM_HEADS"])
BATCH_SIZE=int(config["BATCH_SIZE"])

print("checkpoint test dir",CHECKPOINT_TEST_DIR)
test_features = torch.from_numpy(np.load(TEST_FEATURES_PATH)).float()
test_labels = torch.from_numpy(np.load(TEST_LABELS_PATH)).long()
model_path = os.path.join(CHECKPOINT_TEST_DIR)



# Utility Functions
def load_class_names(file_path):
    class_names = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_names.append(parts[1])
    return class_names
def plot_precision_recall_curves(test_loader, model, device, num_classes):
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs, _ = model(inputs)
            probabilities = nn.Softmax(dim=1)(outputs)
            all_probs.append(probabilities.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)

    nrows, ncols = (num_classes + 11) // 12, 12
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 20))
    axes = axes.flatten()

    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(all_labels == i, all_probs[:, i])
        ap = average_precision_score(all_labels == i, all_probs[:, i])
        axes[i].plot(recall, precision, label=f'AP={ap:.2f}')
        axes[i].set_title(f'Class {i}')
        axes[i].set_xlabel('Recall')
        axes[i].set_ylabel('Precision')
        axes[i].legend()
        axes[i].grid()

    for j in range(num_classes, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
# Model testing function with balanced accuracy
input_size = test_features.shape[-1]
num_classes = len(torch.unique(test_labels))
def test(model_path, test_loader, criterion, device, class_names):
    model = EnhancedBiLSTMWithAttention(input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, num_heads=NUM_HEADS, num_classes=num_classes).to(device)

    # Load the checkpoint weights selectively
    checkpoint = torch.load(model_path)
    model_dict = model.state_dict()

    # Remove keys that don't match
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and model_dict[k].shape == v.shape}

    # Update the model with the compatible weights
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()

    correct, total, running_loss = 0, 0, 0.0
    all_predictions, true_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    test_loss = running_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    balanced_acc = balanced_accuracy_score(true_labels, all_predictions)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    print(f'Balanced Accuracy: {balanced_acc * 100:.2f}%')
    plot_precision_recall_curves(test_loader, model, device, len(class_names))
    report = classification_report(true_labels, all_predictions, target_names=class_names)
    print(report)






# Main
if __name__ == '__main__':
   
    

    # Load class names
    class_names = load_class_names(CLASS_NAMES_PATH)
    print("Test Features Size:", test_features.shape)
    
    # Create test DataLoader
    test_loader = DataLoader(TensorDataset(test_features, test_labels), batch_size=BATCH_SIZE, shuffle=False)
    input_size = test_features.shape[-1]
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Visualize attention weights
    example_inputs, _ = next(iter(test_loader))
    example_inputs = example_inputs.to(device)
    
    # Test the model
    test(model_path, test_loader, criterion, device, class_names)
