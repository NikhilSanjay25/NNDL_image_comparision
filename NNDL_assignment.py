import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights
import matplotlib.pyplot as plt
import numpy as np
import time

# --- 1. Setup and Data Loading ---

print("PyTorch Version:", torch.__version__)
print("Torchvision Version:", torchvision.__version__)

# Set device to GPU (cuda) if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations for the data
# We normalize the data using the mean and std dev of the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

# Download and load the training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

# Download and load the test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# Define class names
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
num_classes = 10

# --- 2. Model 1: CNN from Scratch ---

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # Same architecture as the Keras model
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x

# --- 3. Model 2: Simple Neural Network (MLP) Baseline ---

class MLPBaseline(nn.Module):
    def __init__(self):
        super(MLPBaseline, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(32 * 32 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x

# --- 4. Model 3: Transfer Learning (VGG16) Baseline ---

def create_transfer_learning_model():
    # Load pre-trained VGG16 model
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    
    # Freeze all the layers in the feature extractor
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Replace the classifier with a new one for CIFAR-10
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model

# --- 5. Training and Evaluation Functions ---

def train_model(model, trainloader, criterion, optimizer, epochs=EPOCHS):
    """Function to train the model."""
    model.to(device)
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * correct / total
        
        # Evaluate on test set
        val_loss, val_acc = evaluate_model(model, testloader, criterion)
        
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc / 100.0) # Scale accuracy to 0-1 for plotting
        
        end_time = time.time()
        print(f'Epoch {epoch+1}/{epochs} | Time: {end_time - start_time:.2f}s | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
    print('Finished Training')
    return history


def evaluate_model(model, testloader, criterion):
    """Function to evaluate the model."""
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = running_loss / len(testloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# --- 6. Instantiate, Train, and Evaluate Models ---

# Loss function
criterion = nn.CrossEntropyLoss()

# --- CNN from Scratch ---
print("\n--- Training CNN from Scratch ---")
cnn_model = CustomCNN()
optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)
history_cnn = train_model(cnn_model, trainloader, criterion, optimizer_cnn, epochs=EPOCHS)
final_loss_cnn, final_acc_cnn = evaluate_model(cnn_model, testloader, criterion)

# --- MLP Baseline ---
print("\n--- Training MLP Baseline ---")
mlp_model = MLPBaseline()
optimizer_mlp = optim.Adam(mlp_model.parameters(), lr=LEARNING_RATE)
history_mlp = train_model(mlp_model, trainloader, criterion, optimizer_mlp, epochs=EPOCHS)
final_loss_mlp, final_acc_mlp = evaluate_model(mlp_model, testloader, criterion)

# --- Transfer Learning Model ---
LEARNING_RATE = 0.001  # Assuming this is your rate for models from scratch
EPOCHS = 10

# Add a much smaller learning rate for fine-tuning
LEARNING_RATE_TRANSFER = 0.0001
print("\n--- Training Transfer Learning (VGG16) Baseline ---")
# --- Transfer Learning Model ---
print("\n--- Training Transfer Learning (VGG16) Baseline ---")
transfer_model = create_transfer_learning_model()

# Use the new, smaller learning rate here
optimizer_transfer = optim.Adam(transfer_model.classifier.parameters(), 
                                lr=LEARNING_RATE_TRANSFER) 

history_transfer = train_model(transfer_model, trainloader, criterion, optimizer_transfer, epochs=EPOCHS)
final_loss_transfer, final_acc_transfer = evaluate_model(transfer_model, testloader, criterion)

# --- 7. Final Results and Comparison ---

print("\n--- Final Model Evaluation ---")
print(f"CNN from Scratch       - Test Accuracy: {final_acc_cnn:.2f}%")
print(f"MLP Baseline           - Test Accuracy: {final_acc_mlp:.2f}%")
print(f"Transfer Learning (VGG16) - Test Accuracy: {final_acc_transfer:.2f}%")

# --- Plotting the Results ---

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(16, 6))

# Plot validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history_cnn['val_accuracy'], label='CNN (Scratch)', color='red', linestyle='-')
plt.plot(history_mlp['val_accuracy'], label='MLP Baseline', color='blue', linestyle='--')
plt.plot(history_transfer['val_accuracy'], label='Transfer (VGG16)', color='green', linestyle='-.')
plt.title('Model Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Plot validation loss
plt.subplot(1, 2, 2)
plt.plot(history_cnn['val_loss'], label='CNN (Scratch)', color='red', linestyle='-')
plt.plot(history_mlp['val_loss'], label='MLP Baseline', color='blue', linestyle='--')
plt.plot(history_transfer['val_loss'], label='Transfer (VGG16)', color='green', linestyle='-.')
plt.title('Model Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.suptitle('Model Performance Comparison (PyTorch)', fontsize=16)
plt.show()