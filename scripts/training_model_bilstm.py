import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import os
import gc
import time
import yaml
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, kaiming_uniform_
from sklearn.preprocessing import LabelEncoder
from lstm_model import EnhancedBiLSTMWithAttention
torch.cuda.empty_cache()
gc.collect()
config_path = r'....../configs/config.yaml'
# ---- Load Configurations from config.yaml ----
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
#save checkpoints
CHECKPOINT_OUT_DIR = r'/media/basheer/OVGU/Projects/HAR_Project/cluster/checkpoints'
### DTATSET PATHS
TRAIN_FEATURES_PATH=config['TRAIN_FEATURES_PATH']
TRAIN_LABELS_PATH=config['TRAIN_LABELS_PATH']

VAL_FEATURES_PATH=config['VAL_FEATURES_PATH']
VAL_LABELS_PATH=config['VAL_LABELS_PATH']

TEST_FEATURES_PATH=config['TEST_FEATURES_PATH']
TEST_LABELS_PATH=config['TEST_LABELS_PATH']

# training parameters
LEARNING_RATE=float(config["LEARNING_RATE"])
WEIGHT_DECAY=float(config["WEIGHT_DECAY"])
DROP_OUT=float(config["DROP_OUT"])
HIDDEN_SIZE=int(config["HIDDEN_SIZE"])
NUM_LAYERS=int(config["NUM_LAYERS"])
NUM_HEADS=int(config["NUM_HEADS"])
N_EPOCHS=int(config["N_EPOCHS"])
BATCH_SIZE=int(config["BATCH_SIZE"])
LR_FACTOR=float(config["LR_FACTOR"])
LR_PATIENCE=int(config["LR_PATIENCE"])
NO_FRAMES=int(config["NO_FRAMES"])

# Load your numpy data
train_features = torch.from_numpy(np.load(TRAIN_FEATURES_PATH)).float()
train_labels0 = np.load(TRAIN_LABELS_PATH)
label_encoder = LabelEncoder()
train_labels_numeric = label_encoder.fit_transform(train_labels0)
train_labels = torch.from_numpy(train_labels_numeric).long()

val_features = torch.from_numpy(np.load(VAL_FEATURES_PATH)).float()
val_labels0 = np.load(VAL_LABELS_PATH)
label_encoder = LabelEncoder()
val_labels_numeric = label_encoder.fit_transform(val_labels0)
val_labels = torch.from_numpy(val_labels_numeric).long()

test_features = torch.from_numpy(np.load(TEST_FEATURES_PATH)).float()
test_labels0 = np.load(TEST_LABELS_PATH)
label_encoder = LabelEncoder()
test_labels_numeric = label_encoder.fit_transform(test_labels0)
test_labels = torch.from_numpy(test_labels_numeric).long()
os.makedirs(CHECKPOINT_OUT_DIR, exist_ok=True)






print("run summary ****************************************")
print(f'BATCH_SIZE: {BATCH_SIZE}')
print(f'LEARNING_RATE: {LEARNING_RATE}')
print(f'NUM_EPOCHS: {N_EPOCHS}')
print(f'HIDDEN_SIZE: {HIDDEN_SIZE}')
print(f'DROPOUT_P: {DROP_OUT}')
print(f'PATIENCE: {LR_PATIENCE}')
print(f'NUM_HEADS: {NUM_HEADS}')
print(f'NUM_LAYERS: {NUM_LAYERS}')
print(f'num_frames: {NO_FRAMES}')




# Function to create DataLoader from numpy arrays
def create_data_loader(train_features, train_labels, val_features, val_labels,test_features,test_labels, batch_size):
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = TensorDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader,test_loader




def train(model, train_loader, val_loader, num_epochs, criterion, optimizer, scheduler, device):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    accumulated_time = 0  # Initialize the total time accumulator
    epoch_inference_times = []  # To track per-epoch inference times

    for epoch in range(num_epochs):
        # Print the current epoch number for debugging
        print(f"Starting Epoch {epoch + 1}/{num_epochs}")
        
        start_time = time.time()  # Start timing the epoch
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_inference_times = []  # List to track inference times for this epoch

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # Move to device

            # Measure Inference Time
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            inference_start_time = time.time()

            # Forward pass
            outputs, _ = model(inputs)  # Get outputs and attention weights
            loss = criterion(outputs, labels)

            # Synchronize GPU for accurate timing
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            inference_time = time.time() - inference_start_time

            # Track Inference Time
            epoch_inference_times.append(inference_time / inputs.size(0))  # Per frame

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate average inference time for the epoch
        avg_inference_time = sum(epoch_inference_times) / len(epoch_inference_times)

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # wandb.log({"avg_train_loss": train_loss, "avg_train_accuracy": train_accuracy})

        # Evaluate on validation set
        val_loss, val_accuracy, val_inference_time = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        epoch_time = time.time() - start_time  # Calculate epoch time
        accumulated_time += epoch_time  # Add to accumulated time

        # Print progress for debugging
        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, "
              f"Avg Train Inference Time: {avg_inference_time:.6f} sec, "
              f"Val Inference Time: {val_inference_time:.6f} sec, "
              f"Epoch Time: {epoch_time:.2f} sec, "
              f"Accumulated Time: {accumulated_time:.2f} sec")

        # Step the scheduler
        scheduler.step(val_loss)

        # Save model checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_OUT_DIR, f'bbilstm_checkpoint_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), checkpoint_path)  # Save model state dictionary
            print(f"Model checkpoint saved at {checkpoint_path}")

    # Save the final model
    final_model_path = os.path.join(CHECKPOINT_OUT_DIR, 'bbilstm_final_model.pth')
    torch.save(model.state_dict(), final_model_path)  # Save final model state dictionary
    print(f"Final model saved at {final_model_path}")

    # Print total training time
    print(f"Total training time: {accumulated_time:.2f} seconds")

    return train_losses, train_accuracies, val_losses, val_accuracies


def evaluate(model, data_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    all_preds = []
    all_labels = []
    inference_times = []  # To track inference times in validation

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Measure Inference Time
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            inference_start_time = time.time()

            # Forward pass
            outputs, _ = model(inputs)  # Get outputs and attention weights
            loss = criterion(outputs, labels)

            # Synchronize GPU for accurate timing
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            inference_time = time.time() - inference_start_time
            inference_times.append(inference_time / inputs.size(0))  # Per frame

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate average inference time for validation
    avg_inference_time = sum(inference_times) / len(inference_times)
    loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total

    # wandb.log({"avg_val_loss": loss, "avg_val_accuracy": accuracy})

    return loss, accuracy, avg_inference_time

# Function for testing the best model
input_size = train_features.shape[-1]
num_classes = len(torch.unique(train_labels))

def test(model_path, test_loader, criterion, device):
    # Initialize model architecture
    model = EnhancedBiLSTMWithAttention(input_size, HIDDEN_SIZE, NUM_LAYERS, NUM_HEADS, num_classes).to(device)
    
    # Load the state dictionary
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    
    test_loss, test_accuracy, _ = evaluate(model, test_loader, criterion, device)
    
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    return test_loss, test_accuracy

# Plotting function
def plot_results(train_losses, train_accuracies, val_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # Plotting Losses
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, val_losses, label='val Loss', color='orange')
    plt.title('Losses per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting Accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy', color='blue')
    plt.plot(epochs, val_accuracies, label='val Accuracy', color='orange')
    plt.title('Accuracies per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Function to visualize attention weights


# Main function
def main(train_features, train_labels, val_features, val_labels,test_features,test_labels, num_epochs=N_EPOCHS, batch_size=BATCH_SIZE):
    # Hyperparameters
    input_size = train_features.shape[-1]
    num_classes = len(torch.unique(train_labels))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create DataLoaders
    train_loader, val_loader,test_loader = create_data_loader(train_features, train_labels, val_features, val_labels,test_features,test_labels, batch_size)

    # Initialize model, loss function, optimizer, and scheduler
    model = EnhancedBiLSTMWithAttention(input_size, HIDDEN_SIZE, NUM_LAYERS, NUM_HEADS, num_classes, DROP_OUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        verbose=True
    )

    # Train the model
    train_losses, train_accuracies, val_losses, val_accuracies = train(model, train_loader, val_loader, num_epochs, criterion, optimizer, scheduler, device)

    # Plot results
    plot_results(train_losses, train_accuracies, val_losses, val_accuracies)
    # model_path='/media/basheer/OVGU/Projects/HAR_Project/cluster/checkpoints/bilstm_final_model.pth'
    # test_loss, test_accuracy=test(model_path, test_loader, criterion, device)
    # print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

if __name__ == '__main__':
    # Print sizes of datasets
    print("Training Features Size: ", train_features.shape)
    print("Training Labels Size: ", train_labels.shape)
    print("val Features Size: ", val_features.shape)
    print("val Labels Size: ", val_labels.shape)
    print("test Features Size: ", test_features.shape)
    print("test Labels Size: ", test_labels.shape)

    # Run main function
    main(train_features, train_labels, val_features, val_labels,test_features, test_labels)
