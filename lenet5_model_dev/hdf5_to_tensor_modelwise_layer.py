################################################################
#                Prod Code - All Weights to Single Tensor
################################################################
# Ingests all h5 model weight files in hdf5_dir, takes only model layer weights (91481), flattens by layer and creates list of dictionaries saved as pickle
# Dictionary format is KEYS ('model','weight_tensor_by_layer','weight_tensor_flat')
# model = model name 'church_1' 
# weight_tensor_by_layer = tensor dict by layer of all weights with dimensionality
# weight_tensor_flat = all weights flatten to a single tensor 
# Example: ({'model': 'cassette_player_0',
#            'weight_tensor_by_layer': {'conv2d/sequential/conv2d/bias': tensor([ 0.0049... 1.3019e-01]])},
#            'weight_tensor_flat': tensor([0.0049, 0.0014, 0.0218,  ..., 0.1562, 0.0558, 0.1302])})

import h5py
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import numpy as np
import os
import pickle
from datetime import datetime
import re

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
                               'parameters_by_layer':weight_tensors}
            
            return wght_class_dict
                
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")
        return None
    
def flatten_tensor_allwghts(file_path):
    flat_tensors_lst = []
    weight_tensors = extract_model_weights_from_h5(file_path)
    for k in list(weight_tensors['parameters_by_layer'].keys()):
        flat_tensors_lst.append(weight_tensors['parameters_by_layer'][k].flatten())
    wght_tensor = torch.cat(flat_tensors_lst, dim=0)
    flat_tensor_dict = {'parameters_flat':wght_tensor}

    weight_tensors.update(flat_tensor_dict)

    return weight_tensors

def hdf5_to_pickle(hdf5_dir, output_dir):
    tensor_lst = []
    file_lst=[]
    for f in os.listdir(hdf5_dir):
        file_lst.append(f)
    #file_lst = [f for f in os.listdir(hdf5_dir) if os.path.isfile(os.path.join(dir, f))]
    file_dir_lst = [os.path.join(hdf5_dir, file) for file in file_lst]
    
    for f in file_dir_lst:

        wght_tensor = flatten_tensor_allwghts(f)

        tensor_lst.append(wght_tensor)
    
    dt = datetime.now()
    dt_str = dt.strftime('%Y%m%d_%H%M%S')
    

    file_path = os.path.join(output_dir, f'lenet5_onevall_wghts_bymodel_{dt_str}.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(tensor_lst, f)


if __name__ == '__main__':
    hdf5_dir = '/homes/dkurtenb/projects/HyperGENAI/training_dataset_build/outputs/lenet5_onevall_10000_FINAL/model_weights'
    output_dir = '/homes/dkurtenb/projects/HyperGENAI/training_dataset_build/outputs/lenet5_onevall_10000_FINAL/outputs/pickle_files'
    os.makedirs(output_dir, exist_ok=True)

    tensor_dict = hdf5_to_pickle(hdf5_dir, output_dir) 

