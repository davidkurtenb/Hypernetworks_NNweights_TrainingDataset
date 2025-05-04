import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import pickle
import time
import matplotlib.pyplot as plt
from itertools import product
import json
from functools import partial
import optuna

# Reuse your existing functions for data loading
def unpack_layer_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    key_lst = data[0]['parameters'].keys()
        
    clean_labels = []
    param_tensors_conv2d_lst = []
    param_tensors_conv2d_1_lst = []
    param_tensors_conv2d_2_lst = []
    param_tensors_dense_lst = []
    param_tensors_dense_1_lst = []

    for i in range(len(data)):
        for x in range(len(data[i]['parameters']['conv2d'])):
            label = data[i]['class']
            clean_labels.append(label)

    for i in range(len(data)):
        conv2d_tensor = data[i]['parameters']['conv2d']
        param_tensors_conv2d_lst.append(conv2d_tensor)   

        conv2d_1_tensor = data[i]['parameters']['conv2d_1']
        param_tensors_conv2d_1_lst.append(conv2d_1_tensor) 

        conv2d_2_tensor = data[i]['parameters']['conv2d_2']
        param_tensors_conv2d_2_lst.append(conv2d_2_tensor)        
        
        dense_tensor = data[i]['parameters']['dense']
        param_tensors_dense_lst.append(dense_tensor)

        dense_1_tensor = data[i]['parameters']['dense_1']
        param_tensors_dense_1_lst.append(dense_1_tensor)

    param_tensors_conv2d_lst = np.concatenate(param_tensors_conv2d_lst)
    param_tensors_conv2d_1_lst = np.concatenate(param_tensors_conv2d_1_lst)
    param_tensors_conv2d_2_lst = np.concatenate(param_tensors_conv2d_2_lst)
    param_tensors_dense_lst = np.concatenate(param_tensors_dense_lst)
    param_tensors_dense_1_lst = np.concatenate(param_tensors_dense_1_lst)

    layer_parm_tensors_lst = [param_tensors_conv2d_lst,
                        param_tensors_conv2d_1_lst,
                        param_tensors_conv2d_2_lst,
                        param_tensors_dense_lst,
                        param_tensors_dense_1_lst]

    return key_lst, layer_parm_tensors_lst, clean_labels

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

# Modified MLP architecture with customizable hyperparameters
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
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    tensor_arrays = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            tensor_arrays.append(tensor.numpy().flatten())
        else:
            tensor_arrays.append(np.array(tensor).flatten())
    
    X = np.vstack(tensor_arrays)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)
    
    explained_var = np.sum(pca.explained_variance_ratio_) * 100
    print(f"Total variance explained with {n_components} components: {explained_var:.2f}%")
    
    processed_tensors = [torch.tensor(x, dtype=torch.float32) for x in X_reduced]
    
    return processed_tensors, pca, scaler

def prepare_data(tensors, labels, pca_components=100, batch_size=16, valid_size=0.2, test_size=0.2):
    """Prepare data with preprocessing and splitting"""
    if isinstance(labels, list) and isinstance(labels[0], str):
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
    
    return data_loaders, dataset_sizes, class_names, pca, scaler

def train_model_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch and return statistics"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def validate_model(model, val_loader, criterion, device):
    """Validate the model and return statistics"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def train_model(model, data_loaders, criterion, optimizer, scheduler, 
                device, num_epochs=50, patience=15):
    """Train model with early stopping and learning rate scheduling"""
    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_model_one_epoch(
            model, data_loaders['train'], criterion, optimizer, device
        )
        
        val_loss, val_acc = validate_model(
            model, data_loaders['val'], criterion, device
        )
        
        if scheduler is not None:
            scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model, history, best_val_acc

def evaluate_model(model, test_loader, device, class_names=None):
    """Evaluate the model on test data"""
    from sklearn.metrics import classification_report, confusion_matrix
    
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
    
    if class_names is None:
        class_names = [str(i) for i in range(len(set(all_labels)))]
    
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    
    return report, cm, all_preds, all_labels

def plot_training_history(history, output_path):
    """Plot training/validation loss and accuracy"""
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

def save_confusion_matrix(cm, class_names, output_path):
    """Save confusion matrix as image"""
    import seaborn as sns
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Optuna objective function for hyperparameter optimization
def objective(trial, tensors, labels, pca_components, output_dir):
    # Define hyperparameters to search
    activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu', 'selu', 'gelu'])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.7)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    hidden_layer_sizes = [
        trial.suggest_categorical('hidden_layer1', [64, 128, 256, 512]),
        trial.suggest_categorical('hidden_layer2', [32, 64, 128, 256]),
        trial.suggest_categorical('hidden_layer3', [16, 32, 64, 128])
    ]
    use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
    
    # Prepare data
    data_loaders, dataset_sizes, class_names, pca, scaler = prepare_data(
        tensors, labels, pca_components=pca_components, batch_size=batch_size
    )
    
    # Count number of classes
    if isinstance(labels[0], str):
        num_classes = len(set(labels))
    else:
        num_classes = len(set(labels.numpy() if isinstance(labels, torch.Tensor) else labels))
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedMLP(
        input_size=pca_components, 
        num_classes=num_classes,
        hidden_layers=hidden_layer_sizes,
        dropout_rate=dropout_rate,
        activation=activation,
        use_batch_norm=use_batch_norm
    ).to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    # LR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    # Train model with early stopping
    _, _, best_val_acc = train_model(
        model=model,
        data_loaders=data_loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=30,  # Reduced for hyperparameter search
        patience=10     # Reduced for hyperparameter search
    )
    
    return best_val_acc

def run_hyperparameter_search(tensors, labels, pca_components=100, n_trials=30, output_dir='hyperparameter_search'):
    """Run hyperparameter search using Optuna"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create study
    study = optuna.create_study(direction='maximize')
    
    # Create partial function with fixed arguments
    objective_func = partial(
        objective, 
        tensors=tensors, 
        labels=labels, 
        pca_components=pca_components,
        output_dir=output_dir
    )
    
    # Run optimization
    study.optimize(objective_func, n_trials=n_trials)
    
    # Print best parameters
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best parameters
    best_params = trial.params
    with open(os.path.join(output_dir, 'best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=4)
    
    # Plot optimization history
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(os.path.join(output_dir, 'optimization_history.png'))
    
    # Plot parameter importance
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(os.path.join(output_dir, 'param_importances.png'))
    
    return best_params

def train_with_best_params(tensors, labels, best_params, pca_components=100, output_dir='final_model'):
    """Train final model with best hyperparameters"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract parameters
    activation = best_params['activation']
    dropout_rate = best_params['dropout_rate']
    learning_rate = best_params['learning_rate']
    weight_decay = best_params['weight_decay']
    batch_size = best_params['batch_size']
    hidden_layer_sizes = [
        best_params['hidden_layer1'],
        best_params['hidden_layer2'],
        best_params['hidden_layer3']
    ]
    use_batch_norm = best_params['use_batch_norm']
    
    # Prepare data
    data_loaders, dataset_sizes, class_names, pca, scaler = prepare_data(
        tensors, labels, pca_components=pca_components, batch_size=batch_size
    )
    
    # Count number of classes
    if isinstance(labels[0], str):
        num_classes = len(set(labels))
    else:
        num_classes = len(set(labels.numpy() if isinstance(labels, torch.Tensor) else labels))
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedMLP(
        input_size=pca_components, 
        num_classes=num_classes,
        hidden_layers=hidden_layer_sizes,
        dropout_rate=dropout_rate,
        activation=activation,
        use_batch_norm=use_batch_norm
    ).to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    # LR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Train model
    trained_model, history, _ = train_model(
        model=model,
        data_loaders=data_loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=200,  # Longer for final training
        patience=30      # More patience for final training
    )
    
    # Evaluate model
    report, cm, preds, true_labels = evaluate_model(
        trained_model, data_loaders['test'], device, class_names
    )
    
    # Save results
    with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    # Save model
    torch.save(trained_model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
    
    # Save preprocessors
    with open(os.path.join(output_dir, 'pca_scaler.pkl'), 'wb') as f:
        pickle.dump((pca, scaler), f)
    
    # Plot and save history
    plot_training_history(history, os.path.join(output_dir, 'training_history.png'))
    
    # Plot and save confusion matrix
    save_confusion_matrix(cm, class_names, os.path.join(output_dir, 'confusion_matrix.png'))
    
    return trained_model, report, history, pca, scaler

def main():
    # File paths
    pickle_file_path = '/homes/dkurtenb/projects/HyperGENAI/training_dataset_build/outputs/lenet5_onevall_10000_FINAL/lenet5_onevall_10000_models/pickle_files/lenet5_onevall_wghts_bylayer_class_20250502_102427.pkl'
    base_output_dir = '/homes/dkurtenb/projects/HyperGENAI/training_dataset_build/outputs/lenet5_onevall_10000_FINAL/DNN_classification/layerwise_tuned'
    
    # Load data
    key_lst, layer_parm_tensors_lst, clean_labels = unpack_layer_data(pickle_file_path)
    
    # Convert string labels to numeric indices
    unique_labels = sorted(list(set(clean_labels)))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = [label_to_idx[label] for label in clean_labels]
    
    # Process each layer
    for layer_idx, (layer_tensor, layer_name) in enumerate(zip(layer_parm_tensors_lst, key_lst)):
        print(f"\n{'='*80}")
        print(f"Processing layer: {layer_name}")
        print(f"{'='*80}")
        
        # Create output directories
        layer_output_dir = os.path.join(base_output_dir, layer_name)
        search_output_dir = os.path.join(layer_output_dir, 'hyperparameter_search')
        final_output_dir = os.path.join(layer_output_dir, 'final_model')
        os.makedirs(search_output_dir, exist_ok=True)
        os.makedirs(final_output_dir, exist_ok=True)
        
        # Run hyperparameter search
        print(f"Running hyperparameter search for {layer_name}...")
        best_params = run_hyperparameter_search(
            tensors=layer_tensor,
            labels=numeric_labels,
            pca_components=20,  # Adjust as needed
            n_trials=30,        # Adjust as needed
            output_dir=search_output_dir
        )
        
        # Train final model with best parameters
        print(f"Training final model for {layer_name} with best parameters...")
        trained_model, report, history, pca, scaler = train_with_best_params(
            tensors=layer_tensor,
            labels=numeric_labels,
            best_params=best_params,
            pca_components=20,  # Adjust as needed
            output_dir=final_output_dir
        )
        
        # Print summary
        print(f"\nLayer {layer_name} - Best parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        
        print(f"\nLayer {layer_name} - Test performance:")
        print(f"  Accuracy: {report['accuracy']:.4f}")
        print(f"  Macro F1: {report['macro avg']['f1-score']:.4f}")
        
        print(f"Results saved to {final_output_dir}")

if __name__ == "__main__":
    main()