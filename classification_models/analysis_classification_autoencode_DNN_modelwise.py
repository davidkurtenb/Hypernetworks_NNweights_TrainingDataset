import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import torch
import os


file_path = '/homes/dkurtenb/projects/HyperGENAI/training_dataset_build/outputs/lenet5_onevall_10000_FINAL/outputs/pickle_files/lenet5_onevall_wghts_bymodel_20250502_102356.pkl'

with open(file_path, 'rb') as file:
    data = pickle.load(file)

labels_lst = []
param_tensors_lst = []

for i in range(len(data)):
    label_split = data[i]['model'].split('_')[:-1]
    label = '_'.join(label_split)
    labels_lst.append(label)

    all_flattened = data[i]['parameters_flat']
    param_tensors_lst.append(all_flattened)


class TensorDataset(Dataset):
    """Custom Dataset for loading tensor data"""
    def __init__(self, tensors, labels=None):
        self.tensors = tensors
        self.labels = labels
        
    def __len__(self):
        return len(self.tensors)
    
    def __getitem__(self, idx):
        tensor = self.tensors[idx]
        if self.labels is not None:
            label = self.labels[idx]
            return tensor, label
        return tensor

class Autoencoder(nn.Module):
    """Autoencoder for dimensionality reduction"""
    def __init__(self, input_size, encoding_size):
        super(Autoencoder, self).__init__()
        
        # Calculate intermediate layer sizes
        h1_size = min(input_size // 8, 2048)
        h2_size = min(input_size // 32, 512)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, h1_size),
            nn.BatchNorm1d(h1_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(h1_size, h2_size),
            nn.BatchNorm1d(h2_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(h2_size, encoding_size),
            nn.BatchNorm1d(encoding_size)
        )
        
        # Decoder (for pretraining)
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, h2_size),
            nn.BatchNorm1d(h2_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(h2_size, h1_size),
            nn.BatchNorm1d(h1_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(h1_size, input_size),
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class ClassifierWithAutoencoder(nn.Module):
    def __init__(self, autoencoder, encoding_size, num_classes, dropout_rate=0.5):
        super(ClassifierWithAutoencoder, self).__init__()
        
        # Use pretrained encoder
        self.encoder = autoencoder.encoder
        
        # Freeze encoder weights (optional)
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(encoding_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Get encoded representation
        encoded = self.encoder(x)
        # Classify
        return self.classifier(encoded)

def flatten_tensors(tensors):
    """Flatten tensors and convert to PyTorch tensors"""
    flattened = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            flat = tensor.flatten()
        else:
            flat = torch.tensor(np.array(tensor).flatten(), dtype=torch.float32)
        flattened.append(flat)
    
    return flattened

def train_autoencoder(autoencoder, train_loader, val_loader, num_epochs=100, patience=20):
    """Train autoencoder for dimensionality reduction"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    autoencoder = autoencoder.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # Training phase
        autoencoder.train()
        train_loss = 0.0
        
        for inputs in train_loader:
            if isinstance(inputs, tuple) or isinstance(inputs, list):
                inputs = inputs[0]  # Get only the tensor, not the label
                
            inputs = inputs.to(device)
            
            optimizer.zero_grad()
            _, reconstructed = autoencoder(inputs)
            loss = criterion(reconstructed, inputs)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        # Validation phase
        autoencoder.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs in val_loader:
                if isinstance(inputs, tuple) or isinstance(inputs, list):
                    inputs = inputs[0]  # Get only the tensor, not the label
                    
                inputs = inputs.to(device)
                
                _, reconstructed = autoencoder(inputs)
                loss = criterion(reconstructed, inputs)
                
                val_loss += loss.item() * inputs.size(0)
        
        # Calculate average losses
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Print statistics
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss:.6f}, '
              f'Val Loss: {val_loss:.6f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the model
            os.makedirs('models', exist_ok=True)
            torch.save(autoencoder.state_dict(), 'models/best_autoencoder.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break
    
    # Load best model
    autoencoder.load_state_dict(torch.load('models/best_autoencoder.pth'))
    return autoencoder, history

def train_classifier(model, train_loader, val_loader, num_epochs=50, patience=15, learning_rate=0.0005):
    """Train classifier with encoder + classification head"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_acc = 0.0
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate average losses and accuracies
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        # Update learning rate
        scheduler.step()
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print statistics
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
              f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Early stopping based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save the model
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_classifier.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break
    
    # Load best model
    model.load_state_dict(torch.load('models/best_classifier.pth'))
    return model, history

def evaluate_model(model, test_loader, class_names=None):
    """Evaluate the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Generate classification report
    if class_names is None:
        class_names = [str(i) for i in range(len(set(all_labels)))]
    report = classification_report(all_labels, all_preds, target_names=class_names)
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Create results directory if it doesn't exist
    os.makedirs('/homes/dkurtenb/projects/HyperGENAI/training_dataset_build/outputs/lenet5_onevall_10000_FINAL/autoencode_class', exist_ok=True)
    plt.savefig('/homes/dkurtenb/projects/HyperGENAI/training_dataset_build/outputs/lenet5_onevall_10000_FINAL/autoencode_class/confusion_matrix.png')
    
    print("Classification Report:")
    print(report)
    print("Confusion matrix saved as 'results/confusion_matrix.png'")
    
    return report, all_preds, all_labels

def plot_autoencoder_history(history):
    """Plot autoencoder training history"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Autoencoder Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    
    # Create results directory if it doesn't exist
    os.makedirs('/homes/dkurtenb/projects/HyperGENAI/training_dataset_build/outputs/lenet5_onevall_10000_FINAL/autoencode_class', exist_ok=True)
    plt.savefig('/homes/dkurtenb/projects/HyperGENAI/training_dataset_build/outputs/lenet5_onevall_10000_FINAL/autoencode_class/autoencoder_history.png')
    print("Autoencoder training history saved as 'results/autoencoder_history.png'")

def plot_classifier_history(history):
    """Plot classifier training history"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    # Create results directory if it doesn't exist
    os.makedirs('/homes/dkurtenb/projects/HyperGENAI/training_dataset_build/outputs/lenet5_onevall_10000_FINAL/autoencode_class', exist_ok=True)
    plt.savefig('/homes/dkurtenb/projects/HyperGENAI/training_dataset_build/outputs/lenet5_onevall_10000_FINAL/autoencode_class/classifier_history.png')
    print("Classifier training history saved as 'results/classifier_history.png'")

def visualize_encodings(autoencoder, test_loader, labels, class_names, n_samples=1000):
    """Visualize autoencoder encodings in 2D using t-SNE"""
    from sklearn.manifold import TSNE
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    
    # Get encodings
    encodings = []
    sample_labels = []
    count = 0
    
    with torch.no_grad():
        for inputs in test_loader:
            if isinstance(inputs, tuple) or isinstance(inputs, list):
                # Get tensor and label
                x, label = inputs
                x = x.to(device)
                encoding, _ = autoencoder(x)
                encodings.append(encoding.cpu().numpy())
                sample_labels.extend(label.numpy())
            else:
                # Only tensor, no label
                x = inputs.to(device)
                encoding, _ = autoencoder(x)
                encodings.append(encoding.cpu().numpy())
            
            count += x.size(0)
            if count >= n_samples:
                break
    
    # Combine batches
    encodings = np.vstack(encodings)
    
    # Apply t-SNE
    print("Applying t-SNE for visualization...")
    tsne = TSNE(n_components=2, random_state=42)
    encodings_2d = tsne.fit_transform(encodings[:n_samples])
    
    # Plot
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        mask = np.array(sample_labels[:n_samples]) == i
        plt.scatter(encodings_2d[mask, 0], encodings_2d[mask, 1], label=class_name, alpha=0.7)
    
    plt.legend()
    plt.title('t-SNE visualization of autoencoder encodings')
    plt.tight_layout()
    
    # Create results directory if it doesn't exist
    os.makedirs('/homes/dkurtenb/projects/HyperGENAI/training_dataset_build/outputs/lenet5_onevall_10000_FINAL/autoencode_class', exist_ok=True)
    plt.savefig('/homes/dkurtenb/projects/HyperGENAI/training_dataset_build/outputs/lenet5_onevall_10000_FINAL/autoencode_class/tsne_visualization.png')
    print("t-SNE visualization saved as 'results/tsne_visualization.png'")

def classify_with_autoencoder(tensors, labels, encoding_size=256, batch_size=64, pretrain_epochs=100, finetune_epochs=50):

    # Convert string labels to numeric indices if needed
    if isinstance(labels[0], str):
        print("Converting string labels to numeric indices...")
        unique_labels = sorted(list(set(labels)))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        numeric_labels = [label_to_idx[label] for label in labels]
        labels = numeric_labels
        class_names = unique_labels
    else:
        class_names = [str(i) for i in range(len(set(labels)))]
    
    # Convert labels to tensor
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    
    # Get dataset information
    num_samples = len(tensors)
    num_classes = len(torch.unique(labels))
    print(f"Dataset size: {num_samples} tensors")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    # Flatten tensors
    print("Flattening tensors...")
    flattened_tensors = flatten_tensors(tensors)
    input_size = flattened_tensors[0].shape[0]
    print(f"Input size: {input_size}")
    
    # Split data
    indices = np.arange(num_samples)
    indices_train, indices_temp = train_test_split(
        indices, test_size=0.3, random_state=42, stratify=labels
    )
    indices_val, indices_test = train_test_split(
        indices_temp, test_size=0.5, random_state=42, 
        stratify=labels[indices_temp]
    )
    
    # Create datasets for autoencoder pretraining (no labels needed)
    train_tensors_ae = [flattened_tensors[i] for i in indices_train]
    val_tensors_ae = [flattened_tensors[i] for i in indices_val]
    
    train_dataset_ae = TensorDataset(train_tensors_ae)
    val_dataset_ae = TensorDataset(val_tensors_ae)
    
    # Create datasets for classifier training (with labels)
    train_tensors_clf = [flattened_tensors[i] for i in indices_train]
    val_tensors_clf = [flattened_tensors[i] for i in indices_val]
    test_tensors_clf = [flattened_tensors[i] for i in indices_test]
    
    train_labels_clf = labels[indices_train]
    val_labels_clf = labels[indices_val]
    test_labels_clf = labels[indices_test]
    
    train_dataset_clf = TensorDataset(train_tensors_clf, train_labels_clf)
    val_dataset_clf = TensorDataset(val_tensors_clf, val_labels_clf)
    test_dataset_clf = TensorDataset(test_tensors_clf, test_labels_clf)
    
    # Create dataloaders
    train_loader_ae = DataLoader(train_dataset_ae, batch_size=batch_size, shuffle=True)
    val_loader_ae = DataLoader(val_dataset_ae, batch_size=batch_size)
    
    train_loader_clf = DataLoader(train_dataset_clf, batch_size=batch_size, shuffle=True)
    val_loader_clf = DataLoader(val_dataset_clf, batch_size=batch_size)
    test_loader_clf = DataLoader(test_dataset_clf, batch_size=batch_size)
    
    # Create and train autoencoder
    print("\n----- Stage 1: Training Autoencoder -----")
    autoencoder = Autoencoder(input_size, encoding_size)
    trained_autoencoder, ae_history = train_autoencoder(
        autoencoder, train_loader_ae, val_loader_ae, 
        num_epochs=pretrain_epochs, 
        patience=20
    )
    
    # Plot autoencoder training history
    plot_autoencoder_history(ae_history)
    
    # Visualize encodings (optional)
    try:
        visualize_encodings(trained_autoencoder, test_loader_clf, labels[indices_test], class_names)
    except Exception as e:
        print(f"Visualization error: {str(e)}")
    
    # Create and train classifier with pretrained encoder
    print("\n----- Stage 2: Training Classifier -----")
    classifier = ClassifierWithAutoencoder(trained_autoencoder, encoding_size, num_classes)
    trained_classifier, clf_history = train_classifier(
        classifier, train_loader_clf, val_loader_clf, 
        num_epochs=finetune_epochs, 
        patience=15,
        learning_rate=0.0003
    )
    
    # Plot classifier training history
    plot_classifier_history(clf_history)
    
    # Evaluate model
    print("\n----- Stage 3: Evaluating Model -----")
    report, preds, true_labels = evaluate_model(
        trained_classifier, test_loader_clf, class_names=class_names
    )
    
    return trained_classifier, report, (ae_history, clf_history)


model, report, history = classify_with_autoencoder(
    tensors=param_tensors_lst,
    labels=labels_lst,
    encoding_size=256,        # Size of encoded representation
    batch_size=64,            # Larger batch size for faster training
    pretrain_epochs=100,      # Autoencoder pretraining epochs
    finetune_epochs=500        # Classifier fine-tuning epochs
)