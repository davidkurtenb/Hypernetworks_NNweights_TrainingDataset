import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import time
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import optuna
from functools import partial

def unpack_params(file_path):
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
    
    return labels_lst, param_tensors_lst


class TensorDataset(Dataset):
    """Custom Dataset for loading tensor data"""
    def __init__(self, tensors, labels):
        self.tensors = tensors
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        tensor = self.tensors[idx]
        label = self.labels[idx]
        return tensor, label

class EnhancedMLP(nn.Module):
    """MLP with customizable architecture and regularization"""
    def __init__(self, input_size, num_classes=10, hidden_layers=[256, 128, 64], 
                 dropout_rate=0.5, activation='relu', use_batch_norm=True):
        super(EnhancedMLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Select activation function
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.1)
        elif activation == 'elu':
            act_fn = nn.ELU()
        elif activation == 'selu':
            act_fn = nn.SELU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        else:
            act_fn = nn.ReLU()  # Default
        
        # Build hidden layers
        for h_size in hidden_layers:
            layers.append(nn.Linear(prev_size, h_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_size))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout_rate))
            prev_size = h_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.classifier(x)

def preprocess_with_pca(tensors, n_components=100):
    """Apply PCA for dimensionality reduction"""
    print(f"Original tensor shape: {tensors[0].shape}")
    
    # Convert to numpy arrays if needed
    tensor_arrays = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            tensor_arrays.append(tensor.numpy().flatten())
        else:
            tensor_arrays.append(np.array(tensor).flatten())
    
    # Create a matrix where each row is a flattened tensor
    X = np.vstack(tensor_arrays)
    print(f"Data matrix shape for PCA: {X.shape}")
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    print(f"Reducing dimensions to {n_components} components...")
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)
    
    # Print variance explained
    explained_var = np.sum(pca.explained_variance_ratio_) * 100
    print(f"Total variance explained: {explained_var:.2f}%")
    
    # Convert back to PyTorch tensors
    processed_tensors = [torch.tensor(x, dtype=torch.float32) for x in X_reduced]
    
    return processed_tensors, pca, scaler

def prepare_data(tensors, labels, pca_components=100, batch_size=16, valid_size=0.15, test_size=0.15):
    """Prepare data with preprocessing and splitting"""
    if isinstance(labels[0], str):
        unique_labels = sorted(list(set(labels)))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        numeric_labels = [label_to_idx[label] for label in labels]
        labels = numeric_labels
        class_names = unique_labels
    else:
        class_names = [str(i) for i in range(len(set(labels)))]
    
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    
    processed_tensors, pca, scaler = preprocess_with_pca(tensors, n_components=pca_components)
    
    num_samples = len(processed_tensors)
    indices = np.arange(num_samples)
    
    # First split: train + temp (valid + test)
    indices_train, indices_temp = train_test_split(
        indices, test_size=(valid_size + test_size), random_state=42, stratify=labels
    )
    
    # Second split: valid + test from temp
    test_ratio = test_size / (valid_size + test_size)
    indices_val, indices_test = train_test_split(
        indices_temp, test_size=test_ratio, random_state=42, 
        stratify=labels[indices_temp]
    )
    
    # Extract tensors and labels for each set
    train_tensors = [processed_tensors[i] for i in indices_train]
    val_tensors = [processed_tensors[i] for i in indices_val]
    test_tensors = [processed_tensors[i] for i in indices_test]
    
    train_labels = labels[indices_train]
    val_labels = labels[indices_val]
    test_labels = labels[indices_test]
    
    # Create datasets
    train_dataset = TensorDataset(train_tensors, train_labels)
    val_dataset = TensorDataset(val_tensors, val_labels)
    test_dataset = TensorDataset(test_tensors, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    data_loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset),
        'test': len(test_dataset)
    }
    
    return data_loaders, dataset_sizes, class_names, pca, scaler, labels

def train_and_validate(model, data_loaders, criterion, optimizer, scheduler, device, epoch, log_print=print):
    """Train for one epoch and validate"""
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for inputs, labels in data_loaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    if scheduler is not None:
        scheduler.step()
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for inputs, labels in data_loaders['val']:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    # Calculate average losses and accuracies
    train_loss = train_loss / train_total
    val_loss = val_loss / val_total
    train_acc = train_correct / train_total
    val_acc = val_correct / val_total
    
    # Print statistics
    log_print(f'Epoch {epoch}, '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
          f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    return train_loss, val_loss, train_acc, val_acc

def train_model_with_params(params, data_loaders, input_size, num_classes, device, 
                           num_epochs=50, patience=15, log_print=print):
    """Train a model with specific hyperparameters"""
    # Extract hyperparameters
    activation = params.get('activation', 'relu')
    dropout_rate = params.get('dropout_rate', 0.5)
    learning_rate = params.get('learning_rate', 0.001)
    weight_decay = params.get('weight_decay', 1e-4)
    hidden_layers = params.get('hidden_layers', [256, 128, 64])
    use_batch_norm = params.get('use_batch_norm', True)
    
    # Create model
    model = EnhancedMLP(
        input_size=input_size,
        num_classes=num_classes,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        activation=activation,
        use_batch_norm=use_batch_norm
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        train_loss, val_loss, train_acc, val_acc = train_and_validate(
            model=model,
            data_loaders=data_loaders,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch+1,
            log_print=log_print
        )
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Check if model improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                log_print(f'Early stopping triggered after epoch {epoch+1}')
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, history, best_val_acc

def objective(trial, data_loaders, input_size, num_classes, device, num_epochs=25, patience=10):
    """Optuna objective function for hyperparameter optimization"""
    # Define hyperparameters to search
    params = {
        'activation': trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu', 'selu', 'gelu']),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.7),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'hidden_layers': [
            trial.suggest_categorical('hidden_layer1', [64, 128, 256, 512]),
            trial.suggest_categorical('hidden_layer2', [32, 64, 128, 256]),
            trial.suggest_categorical('hidden_layer3', [16, 32, 64, 128])
        ],
        'use_batch_norm': trial.suggest_categorical('use_batch_norm', [True, False])
    }
    
    # For logging
    def trial_log_print(message):
        print(f"Trial {trial.number}: {message}")
    
    # Train model
    model, history, best_val_acc = train_model_with_params(
        params=params,
        data_loaders=data_loaders,
        input_size=input_size,
        num_classes=num_classes,
        device=device,
        num_epochs=num_epochs,
        patience=patience,
        log_print=trial_log_print
    )
    
    return best_val_acc

def run_hyperparameter_search(data_loaders, input_size, num_classes, n_trials=30, 
                             output_dir='hyperparameter_search'):
    """Run hyperparameter search using Optuna"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Log file
    log_file_path = os.path.join(output_dir, 'hyperparameter_search.log')
    log_file = open(log_file_path, 'w')
    
    def log_print(message):
        print(message)
        log_file.write(message + '\n')
        log_file.flush()
    
    log_print(f"Starting hyperparameter search with {n_trials} trials")
    log_print(f"Input size: {input_size}, Number of classes: {num_classes}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_print(f"Using device: {device}")
    
    # Create study
    study = optuna.create_study(direction='maximize')
    
    # Create partial function with fixed arguments
    obj_func = partial(
        objective, 
        data_loaders=data_loaders, 
        input_size=input_size, 
        num_classes=num_classes, 
        device=device
    )
    
    # Run optimization
    study.optimize(obj_func, n_trials=n_trials)
    
    # Best parameters
    best_trial = study.best_trial
    log_print(f"\nBest trial: {best_trial.number}")
    log_print(f"Best validation accuracy: {best_trial.value:.4f}")
    log_print("Best hyperparameters:")
    for key, value in best_trial.params.items():
        log_print(f"  {key}: {value}")
    
    # Save best parameters
    best_params = best_trial.params
    
    # Convert categorical parameters to proper structure
    best_params['hidden_layers'] = [
        best_params.pop('hidden_layer1'),
        best_params.pop('hidden_layer2'),
        best_params.pop('hidden_layer3')
    ]
    
    with open(os.path.join(output_dir, 'best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=4)
    
    # Save visualization if plotly is available
    try:
        # Optimization history
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(os.path.join(output_dir, 'optimization_history.png'))
        
        # Parameter importance
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image(os.path.join(output_dir, 'param_importances.png'))
        
        # Parallel coordinate plot
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_image(os.path.join(output_dir, 'parallel_coordinate.png'))
        
        log_print("Saved visualization plots")
    except:
        log_print("Warning: Could not generate visualization plots. Make sure plotly is installed.")
    
    log_file.close()
    
    return best_params

def evaluate_model(model, test_loader, class_names, device, output_dir):
    """Evaluate the trained model and save results"""
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
    
    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_text = classification_report(all_labels, all_preds, target_names=class_names)
    
    # Save classification report
    with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report_text)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    print(f"Classification report and confusion matrix saved to {output_dir}")
    print(f"Test accuracy: {report['accuracy']:.4f}")
    
    return report, cm, all_preds, all_labels

def plot_training_history(history, output_path):
    """Plot training and validation metrics"""
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
    plt.savefig(output_path)
    plt.close()

def train_model_final(data_loaders, input_size, num_classes, best_params, 
                     output_dir, num_epochs=300, patience=25):
    """Train the final model with the best hyperparameters"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Log file
    log_file_path = os.path.join(output_dir, 'training_log.txt')
    log_file = open(log_file_path, 'w')
    
    def log_print(message):
        print(message)
        log_file.write(message + '\n')
        log_file.flush()
    
    log_print(f"Training final model with best hyperparameters")
    log_print(f"Parameters: {json.dumps(best_params, indent=2)}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_print(f"Using device: {device}")
    
    # Train model
    model, history, best_val_acc = train_model_with_params(
        params=best_params,
        data_loaders=data_loaders,
        input_size=input_size,
        num_classes=num_classes,
        device=device,
        num_epochs=num_epochs,
        patience=patience,
        log_print=log_print
    )
    
    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
    log_print(f"Saved model with validation accuracy: {best_val_acc:.4f}")
    
    # Plot training history
    plot_training_history(history, os.path.join(output_dir, 'training_history.png'))
    log_print("Saved training history plot")
    
    # Save history data
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    log_file.close()
    
    return model, history

def main():
    # File paths
    pickle_file_path = '/homes/dkurtenb/projects/HyperGENAI/training_dataset_build/outputs/lenet5_onevall_10000_FINAL/lenet5_onevall_10000_models/pickle_files/lenet5_onevall_wghts_bymodel_20250502_102356.pkl'
    base_output_dir = '/homes/dkurtenb/projects/HyperGENAI/training_dataset_build/outputs/lenet5_onevall_10000_FINAL/DNN_classification/modelwise_tuned'
    
    # Create output directories
    search_output_dir = os.path.join(base_output_dir, 'search_results')
    final_output_dir = os.path.join(base_output_dir, 'final_model')
    os.makedirs(search_output_dir, exist_ok=True)
    os.makedirs(final_output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    labels_lst, param_tensors_lst = unpack_params(pickle_file_path)
    
    # Prepare data
    print("Preparing data...")
    pca_components = 20  # Adjust as needed
    batch_size = 32      # Will be optimized later
    data_loaders, dataset_sizes, class_names, pca, scaler, labels = prepare_data(
        tensors=param_tensors_lst,
        labels=labels_lst,
        pca_components=pca_components,
        batch_size=batch_size
    )
    
    # Count number of classes
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    
    # Run hyperparameter search
    print("\nRunning hyperparameter search...")
    best_params = run_hyperparameter_search(
        data_loaders=data_loaders,
        input_size=pca_components,
        num_classes=num_classes,
        n_trials=50,  # Adjust based on computational resources
        output_dir=search_output_dir
    )
    
    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    model, history = train_model_final(
        data_loaders=data_loaders,
        input_size=pca_components,
        num_classes=num_classes,
        best_params=best_params,
        output_dir=final_output_dir,
        num_epochs=500,
        patience=25
    )
    
    # Evaluate final model
    print("\nEvaluating final model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluate_model(
        model=model,
        test_loader=data_loaders['test'],
        class_names=class_names,
        device=device,
        output_dir=final_output_dir
    )
    
    # Save PCA and scaler
    with open(os.path.join(final_output_dir, 'pca_scaler.pkl'), 'wb') as f:
        pickle.dump((pca, scaler), f)
    
    print(f"\nComplete! Results saved to {base_output_dir}")
    print(f"Best hyperparameters: {json.dumps(best_params, indent=2)}")

if __name__ == "__main__":
    main()