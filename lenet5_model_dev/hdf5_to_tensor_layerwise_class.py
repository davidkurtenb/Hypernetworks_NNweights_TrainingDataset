################################################################
#         Prod Code - Layerwise Split to Pickle
################################################################
# Ingests all h5 model weight files in hdf5_dir, takes only model layer weights (91481) by class, flattens and combines weights and corresponding bias 
# Make a list of dictionaries saved as pickle
# One dictionary for each class with length 2
# Dictionary format for first item is KEY = class <str> VALUE = class name <str> example ( 'class': 'cassette_player')
# Second item is KEY = name <str> 'parameters' VALUE = dictionary of parameter biases <dict> example 'parameters': {'layers/dense': tensor([-0.0550, ...0.0738])...'layers/conv2d_2': tensor([-0.0094 ...,-0.0216])}}
# Keys in parameters dict for each weight/bias pair 'layers/dense', 'layers/conv2d', 'layers/conv2d_1', 'layers/dense_1', 'layers/conv2d_2'

import h5py
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import numpy as np
import os
import pickle
from datetime import datetime
import re
from collections import defaultdict

def extract_model_weights_from_h5(h5_file):

    model_nm_split = h5_file.split('/')[-1].split('.')[0].split('_')[2:-1]
    model_key_nm = "_".join(model_nm_split)
    
    weight_tensors = {}
    
    try:
        with h5py.File(h5_file, 'r') as file:
            if 'model_weights' not in file:
                print(f"Error: 'model_weights' not found in {h5_file}")
                return None
                
            def extract_weights(group, prefix=''):
                for key in group.keys():
                    item = group[key]
                    path = f"{prefix}/{key}" if prefix else key
                    
                    if isinstance(item, h5py.Dataset):
                        try:
                            data_array = np.array(item)
                            weight_tensors[path] = torch.from_numpy(data_array)
                            print(f"Converted: {path}, Shape: {data_array.shape}")
                        except Exception as e:
                            print(f"Error converting {path}: {e}")
                    elif isinstance(item, h5py.Group):
                        extract_weights(item, path)
                           
            extract_weights(file['model_weights'])

            
            wght_class_dict = {'model':model_key_nm, 
                               'parameters':weight_tensors}
            
            return wght_class_dict
                
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")
        return None
    
def combine_tensors(input_data):
    import numpy as np
    
    layer_names = ['conv2d',
                   'conv2d_1',
                   'conv2d_2',
                   'dense',
                   'dense_1']

    shape_dict = {
        'conv2d': [(5, 5, 3, 6), (6,)],            
        'conv2d_1': [(5, 5, 6, 16), (16,)],        
        'conv2d_2': [(5, 5, 16, 120), (120,)],     
        'dense': [(480, 84), (84,)],              
        'dense_1': [(84, 1), (1,)]                 
    }

    def identify_base_layer(tensor_shape):
        for layer_name, expected_shapes in shape_dict.items():
            for expected_shape in expected_shapes:
                if tuple(tensor_shape) == expected_shape:
                    return layer_name
        return None

    class_dict = {}
    
    for model in input_data:
        model_name = model['model']
        model_name_parts = model_name.split('_')[:-1]
        class_name = '_'.join(model_name_parts)

        
        if class_name not in class_dict:
            class_dict[class_name] = {
                'class': class_name,
                'parameters': {}
            }
        
        for key, tensor in model['parameters'].items():
            base_layer = key.split('/')[0] 

            if base_layer not in layer_names:
                layer_shape = tuple(tensor.shape)
                identified_layer = identify_base_layer(layer_shape)
                
                if identified_layer:
                    base_layer = identified_layer
                    print(f"Identified layer {base_layer} from shape {layer_shape}")
                else:
                    print(f"WARNING: Could not identify base layer for shape {layer_shape}")
            
            if base_layer not in class_dict[class_name]['parameters']:
                class_dict[class_name]['parameters'][base_layer] = []
            
            class_dict[class_name]['parameters'][base_layer].append({
                'type': 'bias' if '/bias' in key else 'weight',
                'tensor': tensor
            })
    
    for class_name, class_data in class_dict.items():
        for layer_name, tensors in class_data['parameters'].items():
            model_tensors = {}
            
            for tensor_info in tensors:
                tensor_type = tensor_info['type']
                tensor_value = tensor_info['tensor']
                

                if tensor_type == 'bias':
                    model_idx = len([t for t in tensors if t['type'] == 'bias' and id(t['tensor']) < id(tensor_value)])
                    
                    if model_idx not in model_tensors:
                        model_tensors[model_idx] = {}
                    
                    model_tensors[model_idx]['bias'] = tensor_value
                
                elif tensor_type == 'weight':
                    model_idx = len([t for t in tensors if t['type'] == 'weight' and id(t['tensor']) < id(tensor_value)])
                    
                    if model_idx not in model_tensors:
                        model_tensors[model_idx] = {}
                    
                    model_tensors[model_idx]['weight'] = tensor_value
            
            flattened_tensors = []
            
            for model_idx, tensors in model_tensors.items():
                if 'bias' in tensors and 'weight' in tensors:
                    flattened_tensor = np.concatenate([
                        np.array(tensors['bias']).flatten(),
                        np.array(tensors['weight']).flatten()
                    ])
                    
                    flattened_tensors.append(flattened_tensor)
            
            class_data['parameters'][layer_name] = flattened_tensors   

    dt = datetime.now()
    dt_str = dt.strftime('%Y%m%d_%H%M%S')
    
    result = list(class_dict.values())

    file_path = os.path.join(save_dir, f'lenet5_onevall_wghts_bylayer_class_{dt_str}.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(result, f)

    return result,class_dict


hdf5_dir = '/homes/dkurtenb/projects/HyperGENAI/training_dataset_build/outputs/lenet5_onevall_10000_FINAL/model_weights'
file_lst = [f for f in os.listdir(hdf5_dir) if os.path.isfile(os.path.join(hdf5_dir, f))]
file_dir_lst = [os.path.join(hdf5_dir, file) for file in file_lst]

save_dir = '/homes/dkurtenb/projects/HyperGENAI/training_dataset_build/outputs/lenet5_onevall_10000_FINAL/outputs/pickle_files'
os.makedirs(save_dir, exist_ok=True)

wghts_lst = []
for i in range(len(file_dir_lst)):
    wght_dict = extract_model_weights_from_h5(file_dir_lst[i])
    wghts_lst.append(wght_dict)

result,class_dict= combine_tensors(wghts_lst)