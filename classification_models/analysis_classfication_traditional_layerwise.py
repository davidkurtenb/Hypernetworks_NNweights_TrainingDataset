import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import randint, uniform
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

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


def define_param_distributions():
    param_distributions = {
        'Random Forest': {
            'n_estimators': randint(50, 500),
            'max_depth': [None] + list(randint(5, 50).rvs(10)),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        },
        'Naive Bayes': {
            'var_smoothing': uniform(1e-10, 1e-5)
        },
        'SVM': {
            'C': uniform(0.1, 100),
            'kernel': ['rbf', 'linear', 'poly'],
            'gamma': ['scale', 'auto'] + list(uniform(0.001, 1).rvs(5)),
            'degree': [2, 3, 4],
            'probability': [True]
        },
        'XGBoost': {
            'n_estimators': randint(50, 500),
            'max_depth': randint(3, 15),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.5, 0.5),
            'colsample_bytree': uniform(0.5, 0.5),
            'min_child_weight': randint(1, 10),
            'gamma': uniform(0, 0.5),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 1)
        }
    }
    return param_distributions

def compare_classifiers_with_tuning(labels, tensor_data, log_file, n_iter=10, cv=5, n_jobs=-1):
    def log_print(message):
        print(message)
        log_file.write(message + '\n')
        log_file.flush()

    X = np.vstack(tensor_data) if isinstance(tensor_data[0], np.ndarray) else np.array(tensor_data)
    y = np.array(labels)
    
    # Check for class imbalance
    unique_classes, class_counts = np.unique(y, return_counts=True)
    log_print(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
    
    if len(unique_classes) < 2:
        log_print("ERROR: Only one class found in the dataset. Classification requires at least two classes.")
        return None
    
    # Check minimum samples per class to ensure stratified CV works
    min_samples_per_class = np.min(class_counts)
    if min_samples_per_class < cv:
        log_print(f"WARNING: Minimum samples per class ({min_samples_per_class}) is less than CV folds ({cv})")
        log_print(f"Reducing CV folds to {min(3, min_samples_per_class)}")
        cv = max(2, min(3, min_samples_per_class))
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    log_print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    # Use stratified split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Double-check training data has all classes
    train_classes = np.unique(y_train)
    log_print(f"Number of classes in training data: {len(train_classes)}")
    
    if len(train_classes) < 2:
        log_print("ERROR: Training data has only one class after splitting. Increasing training data size.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.1, random_state=42, stratify=y_encoded
        )
        train_classes = np.unique(y_train)
        log_print(f"After adjustment - Classes in training data: {len(train_classes)}")
        
        if len(train_classes) < 2:
            log_print("ERROR: Still only one class in training data. Cannot proceed with classification.")
            return None
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    base_classifiers = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    param_distributions = define_param_distributions()
    
    # Create a stratified k-fold to ensure each fold has examples of each class
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    results = {}
    
    for name, base_classifier in base_classifiers.items():
        log_print(f"\nTuning {name}...")
        
        if name in ['SVM', 'Naive Bayes']:
            X_train_use = X_train_scaled
            X_test_use = X_test_scaled
        else:
            X_train_use = X_train
            X_test_use = X_test
        
        try:
            random_search = RandomizedSearchCV(
                estimator=base_classifier,
                param_distributions=param_distributions[name],
                n_iter=n_iter,
                cv=skf,  # Use stratified k-fold
                n_jobs=n_jobs,
                random_state=42,
                verbose=1,
                scoring='accuracy',
                error_score=np.nan  # Return NaN for failed fits instead of raising error
            )
            
            start_time = time.time()
            random_search.fit(X_train_use, y_train)
            tuning_time = time.time() - start_time
            
            # Check if we have any successful fits
            if np.isnan(random_search.best_score_):
                log_print(f"WARNING: All fits failed for {name}. Skipping this classifier.")
                continue
                
            best_classifier = random_search.best_estimator_
            best_params = random_search.best_params_
            
            start_time = time.time()
            y_pred = best_classifier.predict(X_test_use)
            prediction_time = time.time() - start_time
            
            accuracy = accuracy_score(y_test, y_pred)
            cv_score = random_search.best_score_
            
            y_test_original = label_encoder.inverse_transform(y_test)
            y_pred_original = label_encoder.inverse_transform(y_pred)
            
            results[name] = {
                'best_params': best_params,
                'best_cv_score': cv_score,
                'test_accuracy': accuracy,
                'tuning_time': tuning_time,
                'prediction_time': prediction_time,
                'classification_report': classification_report(y_test_original, y_pred_original),
                'all_cv_results': pd.DataFrame(random_search.cv_results_)
            }
            
            log_print(f"{name} tuning completed in {tuning_time:.2f} seconds")
            log_print(f"Best CV Score: {cv_score:.4f}")
            log_print(f"Test Accuracy: {accuracy:.4f}")
            log_print(f"Best Parameters: {best_params}")
            
        except Exception as e:
            log_print(f"ERROR: Failed to tune {name}: {str(e)}")
            log_print("Continuing with other classifiers...")
    
    if not results:
        log_print("WARNING: No successful classifier tuning. Please check your data.")
    
    return results

def print_comparison_results(results, log_file):
    def log_print(message):
        print(message)
        log_file.write(message + '\n')
        log_file.flush()
    
    if not results:
        log_print("No results to compare.")
        return None
    
    summary_data = []
    for classifier, metrics in results.items():
        summary_data.append({
            'Classifier': classifier,
            'Best CV Score': f"{metrics['best_cv_score']:.4f}",
            'Test Accuracy': f"{metrics['test_accuracy']:.4f}",
            'Tuning Time': f"{metrics['tuning_time']:.2f}s",
            'Prediction Time': f"{metrics['prediction_time']:.4f}s"
        })
    
    df_summary = pd.DataFrame(summary_data)
    log_print("\n" + "="*80)
    log_print("CLASSIFIER COMPARISON SUMMARY (WITH HYPERPARAMETER TUNING)")
    log_print("="*80)
    log_print(df_summary.to_string(index=False))
    log_print("="*80)
    
    log_print("\nBEST PARAMETERS FOR EACH CLASSIFIER:")
    log_print("-" * 50)
    for classifier, metrics in results.items():
        log_print(f"\n{classifier}:")
        for param, value in metrics['best_params'].items():
            log_print(f"  {param}: {value}")
    
    best_cv = max(results.items(), key=lambda x: x[1]['best_cv_score'])
    best_test = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    
    log_print("\n" + "="*50)
    log_print(f"Best CV performance: {best_cv[0]} ({best_cv[1]['best_cv_score']:.4f})")
    log_print(f"Best test accuracy: {best_test[0]} ({best_test[1]['test_accuracy']:.4f})")
    log_print("="*50)
    
    return df_summary

def plot_hyperparameter_importance(results, output_dir, layer_name, top_n=3):
    if not results:
        print(f"No results to plot for layer {layer_name}")
        return
    
    for classifier, metrics in results.items():
        try:
            cv_results = metrics['all_cv_results']
            
            param_cols = [col for col in cv_results.columns if col.startswith('param_')]
            
            if param_cols:
                plt.figure(figsize=(12, 8))
                
                plot_count = 0
                for idx, param_col in enumerate(param_cols[:top_n]):
                    if idx >= 4: 
                        break
                    
                    # Skip params with all the same value
                    if cv_results[param_col].nunique() <= 1:
                        continue
                        
                    plot_count += 1
                    plt.subplot(2, 2, plot_count)
                    param_name = param_col.replace('param_', '')
                    
                    # Handle different parameter types
                    if cv_results[param_col].dtype == np.float64 or cv_results[param_col].dtype == np.int64:
                        # For numeric parameters with many values, bin them
                        if cv_results[param_col].nunique() > 10:
                            try:
                                cv_results[f'{param_col}_binned'] = pd.qcut(
                                    cv_results[param_col], 
                                    q=min(5, cv_results[param_col].nunique()),
                                    duplicates='drop'
                                )
                                sns.boxplot(
                                    data=cv_results, 
                                    x=f'{param_col}_binned', 
                                    y='mean_test_score'
                                )
                            except Exception as e:
                                print(f"Error binning parameter {param_name}: {str(e)}")
                                sns.scatterplot(
                                    data=cv_results, 
                                    x=param_col, 
                                    y='mean_test_score'
                                )
                        else:
                            # For numeric parameters with few values, use them directly
                            sns.boxplot(
                                data=cv_results,
                                x=param_col,
                                y='mean_test_score',
                                order=sorted(cv_results[param_col].unique())
                            )
                    else:
                        # For categorical parameters
                        median_scores = cv_results.groupby(param_col)['mean_test_score'].median().sort_values(ascending=False)
                        
                        if len(median_scores) > 0:
                            sns.boxplot(
                                data=cv_results, 
                                x=param_col, 
                                y='mean_test_score',
                                order=median_scores.index
                            )
                    
                    plt.title(f'{param_name} Impact on CV Score')
                    plt.xlabel(param_name)
                    plt.ylabel('CV Score')
                    plt.xticks(rotation=45)
                
                if plot_count > 0:
                    plt.tight_layout()
                    filename = f'{layer_name}_{classifier.lower().replace(" ", "_")}_hyperparameter_impact.png'
                    plt.savefig(os.path.join(output_dir, filename))
                plt.close()
        except Exception as e:
            print(f"Error plotting hyperparameter importance for {classifier} on layer {layer_name}: {str(e)}")

def check_data_characteristics(tensors, labels, log_file):
    """Analyze data characteristics to identify potential issues"""
    def log_print(message):
        print(message)
        log_file.write(message + '\n')
        log_file.flush()
    
    log_print("\n" + "="*50)
    log_print("DATA CHARACTERISTICS ANALYSIS")
    log_print("="*50)
    
    # Check tensor shapes and sizes
    log_print(f"Number of samples: {len(tensors)}")
    log_print(f"Tensor shape: {tensors[0].shape}")
    flattened = np.array([t.flatten() for t in tensors])
    log_print(f"Flattened feature dimensions: {flattened.shape[1]}")
    
    # Check class distribution
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    class_dist = dict(zip(unique_classes, class_counts))
    log_print(f"Number of classes: {len(unique_classes)}")
    log_print(f"Class distribution: {class_dist}")
    
    # Check for extreme imbalance
    if len(unique_classes) > 1:
        min_class = min(class_counts)
        max_class = max(class_counts)
        imbalance_ratio = max_class / min_class
        log_print(f"Class imbalance ratio (max/min): {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 10:
            log_print("WARNING: Severe class imbalance detected. Consider resampling or using class weights.")
    
    # Check for outliers in tensor values
    tensor_array = np.vstack(tensors)
    tensor_mean = np.mean(tensor_array)
    tensor_std = np.std(tensor_array)
    tensor_min = np.min(tensor_array)
    tensor_max = np.max(tensor_array)
    
    log_print(f"Tensor value statistics:")
    log_print(f"  Mean: {tensor_mean:.4f}")
    log_print(f"  Std: {tensor_std:.4f}")
    log_print(f"  Min: {tensor_min:.4f}")
    log_print(f"  Max: {tensor_max:.4f}")
    
    # Check for zero variance features
    variances = np.var(tensor_array, axis=0)
    zero_var_count = np.sum(variances < 1e-10)
    if zero_var_count > 0:
        log_print(f"WARNING: {zero_var_count} features have near-zero variance")
    
    log_print("="*50)

def main():
    # File paths
    lenet_params_layer = '/homes/dkurtenb/projects/HyperGENAI/training_dataset_build/outputs/lenet5_onevall_10000_FINAL/lenet5_onevall_10000_models/pickle_files/lenet5_onevall_wghts_bylayer_class_20250502_102427.pkl'
    output_dir = '/homes/dkurtenb/projects/HyperGENAI/training_dataset_build/outputs/lenet5_onevall_10000_FINAL/stats_classification/layerwise'
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    key_lst, layer_parm_tensors_lst, clean_labels = unpack_layer_data(lenet_params_layer)

    # Convert labels to numeric indices
    unique_labels = sorted(list(set(clean_labels)))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = [label_to_idx[label] for label in clean_labels]

    # Process each layer
    for idx, (tensor_data, layer_key) in enumerate(zip(layer_parm_tensors_lst, key_lst)):
        print(f"\n{'='*20} Processing Layer: {layer_key} {'='*20}")
        
        layer_output_dir = os.path.join(output_dir, layer_key)
        os.makedirs(layer_output_dir, exist_ok=True)
        
        layer_log_file = open(os.path.join(layer_output_dir, f'analysis_report.txt'), 'w')
        
        # Analyze data characteristics
        check_data_characteristics(tensor_data, numeric_labels, layer_log_file)
        
        # Determine appropriate sample size
        # Start with a smaller sample, increase if needed
        n_samples = min(500, len(tensor_data))
        
        # Ensure at least 2 samples per class
        unique_classes, class_counts = np.unique(numeric_labels[:n_samples], return_counts=True)
        while len(unique_classes) < 2 or min(class_counts) < 2:
            n_samples = min(n_samples + 100, len(tensor_data))
            if n_samples == len(tensor_data):
                break
            unique_classes, class_counts = np.unique(numeric_labels[:n_samples], return_counts=True)
        
        log_print = lambda msg: (print(msg), layer_log_file.write(msg + '\n'), layer_log_file.flush())
        log_print(f"Using {n_samples} samples for analysis")
        
        # If we still don't have at least 2 classes, skip this layer
        if len(unique_classes) < 2:
            log_print(f"ERROR: Only one class found in layer {layer_key}. Skipping classification.")
            continue
            
        # Run classifier comparison
        try:
            results = compare_classifiers_with_tuning(
                numeric_labels[:n_samples], 
                tensor_data[:n_samples], 
                layer_log_file, 
                n_iter=25,  
                cv=3
            )
            
            if results:
                summary_df = print_comparison_results(results, layer_log_file)
                if summary_df is not None:
                    summary_df.to_csv(os.path.join(layer_output_dir, f'classification_model_results.csv'))
                
                # Plot hyperparameter importance
                plot_hyperparameter_importance(results, layer_output_dir, layer_key)
            else:
                log_print(f"No results available for layer {layer_key}. Skipping visualization.")
                
        except Exception as e:
            log_print(f"ERROR: Failed to process layer {layer_key}: {str(e)}")
            
        layer_log_file.close()
        print(f"Completed processing for layer: {layer_key}")
    
    print("\nAll layers processed successfully!")

if __name__ == "__main__":
    main()