import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
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
            'C': uniform(0.01, 100),
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto'] + list(uniform(0.001, 1).rvs(10)),
            'degree': randint(2, 5),  # for polynomial kernel
            'probability': [True]  # needed for probability estimates
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
    
    # Encode string labels to integers for XGBoost
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    log_print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
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
    
    results = {}
    
    for name, base_classifier in base_classifiers.items():
        log_print(f"\nTuning {name}...")
        
        if name in ['SVM', 'Naive Bayes']:
            X_train_use = X_train_scaled
            X_test_use = X_test_scaled
        else:
            X_train_use = X_train
            X_test_use = X_test
        
        random_search = RandomizedSearchCV(
            estimator=base_classifier,
            param_distributions=param_distributions[name],
            n_iter=n_iter,
            cv=cv,
            n_jobs=n_jobs,
            random_state=42,
            verbose=1,
            scoring='accuracy'
        )
        
        start_time = time.time()
        random_search.fit(X_train_use, y_train)
        tuning_time = time.time() - start_time
        
        best_classifier = random_search.best_estimator_
        best_params = random_search.best_params_
        
        start_time = time.time()
        y_pred = best_classifier.predict(X_test_use)
        prediction_time = time.time() - start_time
        
        accuracy = accuracy_score(y_test, y_pred)
        cv_score = random_search.best_score_
        
        # Convert numerical predictions back to original labels for reporting
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
    
    return results

def print_comparison_results(results,log_file):
    
    def log_print(message):
        print(message)
        log_file.write(message + '\n')
        log_file.flush()
    
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

def plot_hyperparameter_importance(results, output_dir, top_n=3):
    
    for classifier, metrics in results.items():
        cv_results = metrics['all_cv_results']
        
        param_cols = [col for col in cv_results.columns if col.startswith('param_')]
        
        if param_cols:
            plt.figure(figsize=(12, 8))
            
            for idx, param_col in enumerate(param_cols[:top_n]):
                if idx >= 4:  # subplot limit
                    break
                    
                plt.subplot(2, 2, idx + 1)
                param_name = param_col.replace('param_', '')
                
                unique_values = cv_results[param_col].value_counts()
                if len(unique_values) > 10:  # Too many unique values, create bins
                    cv_results[param_col] = pd.qcut(cv_results[param_col], q=5, duplicates='drop')
                
                sns.boxplot(data=cv_results, 
                           x=param_col, 
                           y='mean_test_score',
                           order=cv_results.groupby(param_col)['mean_test_score'].median().sort_values(ascending=False).index)
                
                plt.title(f'{param_name} Impact on CV Score')
                plt.xlabel(param_name)
                plt.ylabel('CV Score')
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{classifier.lower().replace(" ", "_")}_hyperparameter_impact.png'))
            plt.close()


##############################################################
# MAIN
#############################################

if __name__ == "__main__":
    
    lenet_params_model = '/homes/dkurtenb/projects/HyperGENAI/training_dataset_build/outputs/lenet5_onevall_10000_FINAL/lenet5_onevall_10000_models/pickle_files/lenet5_onevall_wghts_bymodel_20250502_102356.pkl'
    
    labels_lst, param_tensors_lst = unpack_params(lenet_params_model)

    output_dir = '/homes/dkurtenb/projects/HyperGENAI/training_dataset_build/outputs/lenet5_onevall_10000_FINAL/stats_classification/modelwise'
    os.makedirs(output_dir,exist_ok=True)

    log_file = open(os.path.join(output_dir, f'analysis_report.txt'), 'w')

    def log_print(message):
        print(message)
        log_file.write(message + '\n')
        log_file.flush()
        
    log_print(f"Running comparison with hyperparameter tuning...")
    
    # Run comparison with fewer iterations for demo (increase n_iter for real data)
    results = compare_classifiers_with_tuning(labels_lst, param_tensors_lst, log_file, n_iter=20, cv=3)
    #results = compare_classifiers_with_tuning(labels_lst, param_tensors_lst, log_file, n_iter=5, cv=3)
    
    # Print results
    summary_df = print_comparison_results(results, log_file)
    summary_df.to_csv(os.path.join(output_dir,'classification_model_tradML_results.csv'))
    # Generate hyperparameter impact plots
    plot_hyperparameter_importance(results, output_dir)

