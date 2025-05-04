##############################################################################
#                PROD CODE
###############################################################################

import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import os 
import tensorflow_datasets as tfds
from datetime import datetime
import time

print(f"TensorFlow version: {tf.__version__}")
print(f"Eager execution: {tf.executing_eagerly()}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

tf.debugging.set_log_device_placement(True)

# Set HDF5 environment variables for better compatibility
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

#configure GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("No GPUs found. Running on CPU.")

try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print(f"Compute dtype: {tf.keras.mixed_precision.global_policy().compute_dtype}")
    print(f"Variable dtype: {tf.keras.mixed_precision.global_policy().variable_dtype}")
except Exception as e:
    print(f"Error setting mixed precision: {e}")
    print("Continuing with default precision")

max_count = 1000
batch_size = 128
epochs = 20
learning_rate = 0.001

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
output_base_dir = f'/homes/dkurtenb/projects/HyperGENAI/training_dataset_build/outputs/lenet5_onevall_10000/run_{current_time}'

plot_dir = os.path.join(output_base_dir, 'plots')
model_dir = os.path.join(output_base_dir, 'model_weights')
savedmodel_dir = os.path.join(output_base_dir, 'saved_models')
weights_dir = os.path.join(output_base_dir, 'weight_files')
checkpoint_dir = os.path.join(output_base_dir, 'checkpoints')

for directory in [plot_dir, model_dir, savedmodel_dir, weights_dir, checkpoint_dir]:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")

#data
print("Loading dataset...")
dataset_name = 'imagenette/320px-v2' 
dataset, info = tfds.load(dataset_name, with_info=True, as_supervised=True)
print("Dataset loaded successfully!")

#Class labels
class_labels = {0: 'tench',
                1: 'english_springer',
                2: 'cassette_player',
                3: 'chain_saw',
                4: 'church',
                5: 'french_horn',
                6: 'garbage_truck',
                7: 'gas_pump',
                8: 'golf_ball',
                9: 'parachute'}

#metrics
accuracy_dict = {}
precision_dict = {}
recall_dict = {}
f1_dict = {}
auc_dict = {}

#build LeNet-5 model
def build_lenet5():
    model = models.Sequential()
    model.add(layers.Conv2D(6, kernel_size=(5, 5), activation='tanh', input_shape=(32, 32, 3), padding='same'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(16, kernel_size=(5, 5), activation='tanh'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(120, kernel_size=(5, 5), activation='tanh'))
    model.add(layers.Flatten())
    model.add(layers.Dense(84, activation='tanh'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification
    return model

# Function to save model in multiple formats with verification
def save_model_robust(model, base_path, class_name, count):
    """Save model in multiple formats with verification to ensure at least one works"""
    
    success = False
    errors = []
    
    # Format 1: SavedModel format (most reliable)
    try:
        savedmodel_path = os.path.join(savedmodel_dir, f'binary_oneVall_{class_name}_{count}_model')
        print(f"Saving model in SavedModel format to {savedmodel_path}")
        model.save(savedmodel_path)
        if os.path.exists(savedmodel_path):
            print(f"? SavedModel successfully saved at {savedmodel_path}")
            # Verify saved model can be loaded
            try:
                test_load = tf.keras.models.load_model(savedmodel_path)
                print(f"? SavedModel successfully verified by loading")
                success = True
            except Exception as e:
                print(f"? SavedModel verification failed: {e}")
                errors.append(f"SavedModel verification error: {str(e)}")
        else:
            print(f"? SavedModel directory not created")
            errors.append("SavedModel directory not created")
    except Exception as e:
        print(f"? Error saving SavedModel format: {e}")
        errors.append(f"SavedModel error: {str(e)}")
    
    # Format 2: H5 format (with file sync after save)
    try:
        h5_path = os.path.join(model_dir, f'binary_oneVall_{class_name}_{count}_model.h5')
        print(f"Saving model in H5 format to {h5_path}")
        model.save(h5_path, save_format='h5')
        
        # Force file system sync
        try:
            os.system('sync')
            time.sleep(2)  # Wait for sync to complete
        except:
            pass
            
        if os.path.exists(h5_path):
            size = os.path.getsize(h5_path)
            print(f"? H5 model saved at {h5_path} (size: {size/1024:.2f} KB)")
            if size > 1000:  # Check if file size is reasonable
                success = True
            else:
                print(f"? H5 file may be empty or corrupt (size: {size} bytes)")
                errors.append(f"H5 file too small: {size} bytes")
        else:
            print(f"? H5 file not created")
            errors.append("H5 file not created")
    except Exception as e:
        print(f"? Error saving H5 format: {e}")
        errors.append(f"H5 error: {str(e)}")
    
    # Format 3: Manual weights saving
    try:
        weights_path = os.path.join(weights_dir, f'binary_oneVall_{class_name}_{count}_weights.h5')
        print(f"Saving weights to {weights_path}")
        model.save_weights(weights_path)
        
        # Force file system sync
        try:
            os.system('sync')
            time.sleep(2)  # Wait for sync to complete
        except:
            pass
            
        if os.path.exists(weights_path):
            size = os.path.getsize(weights_path)
            print(f"? Weights saved at {weights_path} (size: {size/1024:.2f} KB)")
            if size > 1000:  # Check if file size is reasonable
                success = True
            else:
                print(f"? Weights file may be empty or corrupt (size: {size} bytes)")
                errors.append(f"Weights file too small: {size} bytes")
        else:
            print(f"? Weights file not created")
            errors.append("Weights file not created")
    except Exception as e:
        print(f"? Error saving weights: {e}")
        errors.append(f"Weights error: {str(e)}")
    
    # Try alternative approach: save weights as TF checkpoint
    try:
        checkpoint_path = os.path.join(checkpoint_dir, f'binary_oneVall_{class_name}_{count}')
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.save(checkpoint_path)
        print(f"? TF Checkpoint saved at {checkpoint_path}")
        success = True
    except Exception as e:
        print(f"? Error saving TF checkpoint: {e}")
        errors.append(f"Checkpoint error: {str(e)}")
    
    # Return status
    if success:
        print(f"? Model saving completed successfully in at least one format")
        return True
    else:
        print(f"? ALL MODEL SAVING ATTEMPTS FAILED")
        print(f"Errors encountered: {errors}")
        return False

csv_filename = os.path.join(output_base_dir, 'model_performance_metrics.csv')
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Class', 'Count', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'])

#train
count = 0
start_time = datetime.now()
print(f"Starting training at {start_time}")

try:
    print("Testing GPU computation...")
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
        print(f"Matrix multiplication result: {c}")
    print("GPU computation test successful!")
except Exception as e:
    print(f"GPU computation test failed: {e}")
    print("This may indicate a problem with GPU access or configuration.")

while count < max_count:
    for k, v in zip(class_labels.keys(), class_labels.values()):
        print(f"COUNT: {count} - CLASS: {v}")

        train_dataset = dataset['train']
        validation_dataset = dataset['validation']

        def map_labels_one_vs_all(image, label):
            positive_class = k 
            binary_label = tf.where(tf.equal(label, positive_class), 1, 0)  
            return image, binary_label

        train_dataset = train_dataset.map(map_labels_one_vs_all)
        validation_dataset = validation_dataset.map(map_labels_one_vs_all)

        def preprocess(image, label):
            image = tf.image.resize(image, [32, 32])  
            image = tf.cast(image, tf.float32) / 255.0 
            return image, label

        train_dataset = train_dataset.map(preprocess).shuffle(1024).batch(batch_size)
        validation_dataset = validation_dataset.map(preprocess).batch(batch_size)

        model = build_lenet5()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='binary_crossentropy', metrics=['accuracy'])

        # Custom callback for more reliable checkpointing
        class RobustCheckpoint(tf.keras.callbacks.Callback):
            def __init__(self, filepath, monitor='val_loss', verbose=1, save_best_only=True):
                super(RobustCheckpoint, self).__init__()
                self.filepath = filepath
                self.monitor = monitor
                self.verbose = verbose
                self.save_best_only = save_best_only
                self.best = float('inf') if 'loss' in monitor else -float('inf')
                
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                current = logs.get(self.monitor)
                
                if current is None:
                    return
                    
                if self.save_best_only:
                    if ('loss' in self.monitor and current < self.best) or \
                       ('loss' not in self.monitor and current > self.best):
                        if self.verbose > 0:
                            print(f'\nEpoch {epoch+1}: {self.monitor} improved from {self.best:.5f} to {current:.5f}, saving model')
                        self.best = current
                        
                        # Try multiple saving methods
                        try:
                            self.model.save_weights(f"{self.filepath}_epoch{epoch+1}.h5")
                            print(f"? Successfully saved weights to {self.filepath}_epoch{epoch+1}.h5")
                        except Exception as e:
                            print(f"? Failed to save weights: {e}")
                            
                        try:
                            checkpoint = tf.train.Checkpoint(model=self.model)
                            checkpoint.save(f"{self.filepath}_epoch{epoch+1}_ckpt")
                            print(f"? Successfully saved checkpoint to {self.filepath}_epoch{epoch+1}_ckpt")
                        except Exception as e:
                            print(f"? Failed to save checkpoint: {e}")
                else:
                    if self.verbose > 0:
                        print(f'\nEpoch {epoch+1}: saving model to {self.filepath}')
                    try:
                        self.model.save_weights(f"{self.filepath}_epoch{epoch+1}.h5")
                    except Exception as e:
                        print(f"Failed to save weights: {e}")

        # Create robust checkpoint callback
        robust_checkpoint = RobustCheckpoint(
            filepath=os.path.join(checkpoint_dir, f'binary_oneVall_{v}_{count}'),
            monitor='val_loss',
            verbose=1,
            save_best_only=True
        )

        # Train the model
        history = model.fit(
            train_dataset, 
            epochs=epochs, 
            validation_data=validation_dataset,
            callbacks=[robust_checkpoint],
            verbose=1
        )

        # Save model using our robust function
        save_success = save_model_robust(model, output_base_dir, v, count)
        if not save_success:
            print("WARNING: Failed to save model in any format. Will attempt to continue training.")

        # Evaluation
        y_true = []
        y_pred = []
        y_pred_proba = []

        print(f"Evaluating model for class {v} (count {count})...")
        
        for images, labels in validation_dataset:
            y_true.extend(labels.numpy())  
            y_pred_proba_batch = model.predict(images, verbose=0)  
            y_pred.extend(y_pred_proba_batch.round())  
            y_pred_proba.extend(y_pred_proba_batch)  

        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        y_pred_proba = np.array(y_pred_proba).flatten()

        #metrics
        accuracy = np.mean(y_true == y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc_score = roc_auc_score(y_true, y_pred_proba)

        accuracy_dict[v] = accuracy
        precision_dict[v] = precision
        recall_dict[v] = recall
        f1_dict[v] = f1
        auc_dict[v] = auc_score

        print(f"Metrics for class: {v} count: {count}: Accuracy={accuracy:.2f}, Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}, AUC={auc_score:.2f}")

        nested_plot_dir = os.path.join(plot_dir, f'{v}_{count}')
        os.makedirs(nested_plot_dir, exist_ok=True) 

        #ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Class {v}')
        plt.legend(loc="lower right")

        plot_filename = os.path.join(nested_plot_dir, f'roc_curve_class_{v}_count.png')
        plt.savefig(plot_filename)  
        plt.close()  
        
        #accuracy and loss plots
        plt.figure(figsize=(12, 5))
        
        #accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'Model Accuracy for Class {v}', fontsize=10)
        plt.xlabel('Epochs', fontsize=8)
        plt.ylabel('Accuracy', fontsize=8)
        plt.legend(loc='lower right')

        #loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Model Loss for Class {v}', fontsize=10)
        plt.xlabel('Epochs', fontsize=8)
        plt.ylabel('Loss', fontsize=8)
        plt.legend(loc='upper right')

        plt.tight_layout()
        acl_plot_filename = os.path.join(nested_plot_dir, f'accuracy_loss_{v}.png')
        plt.savefig(acl_plot_filename) 
        plt.close()  

        # Clear session to release GPU memory
        tf.keras.backend.clear_session()

        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([v, count, accuracy, precision, recall, f1, auc_score])

    count += 1

    if count % 1 == 0:  
        elapsed = datetime.now() - start_time
        eta = (elapsed / count) * (max_count - count) if count > 0 else datetime.timedelta(0)
        print(f"Completed {count}/{max_count} iterations in {elapsed}")
        print(f"Estimated time remaining: {eta}")

end_time = datetime.now()
total_time = end_time - start_time
print(f"Training completed at {end_time}")
print(f"Total training time: {total_time}")