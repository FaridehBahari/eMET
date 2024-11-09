#################################
#         TODO: preds are count not rate

from collections import namedtuple
from readFtrs_Rspns import create_TestTrain_TwoSources, get_latest_commit_hash, read_response
import os
import ast
import pickle
import pandas as pd
import numpy as np
from readFtrs_Rspns import scale_train, scale_test, load_data
from performance.assessModels import assess_model
from models.repeated_train_test import save_metrics_summary
import shutil
import configparser
from simulation_settings import load_sim_settings, config_get
import platform
   


def fit_model(X_train, Y_train, X_test, Y_test, run_func, predict_func,
              make_pred = True, *args):
    
    model = run_func(X_train, Y_train, args[0])
    length_test_elems = Y_test['length']
    length_train_elems = Y_train['length']
    
    if make_pred:
        predRates_test = predict_func(model, X_test, length_test_elems)
        predRates_train = predict_func(model, X_train, length_train_elems)
    else:
        predRates_train = predRates_test = pd.DataFrame({'predRate': None}, 
                                                        index=['None'])
    
    
    # create a named tuple
    DataTuple = namedtuple('DataTuple', ['model', 'predRates_train', 
                                         'predRates_test', 'run_func',
                                         'predict_func'])
    
    fitted_Model = DataTuple(model=model, predRates_train=predRates_train,
                     predRates_test = predRates_test, run_func = run_func,
                     predict_func = predict_func)
    
    return  fitted_Model


        
def write_readme_file(information, filename):
    # Call the function to get the latest commit hash
    latest_commit_hash = get_latest_commit_hash()
    information['git_commit']= latest_commit_hash
    with open(filename, 'w') as f:
        # Write the dictionary to the file
        f.write('information\n\n')
        for key, value in information.items():
            f.write(f'- {key}: {value}\n')
            


def RUN_BMR(sim_setting,  X_train, Y_train, X_test, Y_test, make_pred = True,
            overwrite = True):
    
    models = sim_setting['models']
    base_dir = sim_setting['base_dir']
        
    for key in models:
       
        m = models[key]
        name = m['save_name']
        
        os.makedirs(f'{base_dir}/{name}/', exist_ok= True)
        readme_file_name = f'{base_dir}/{name}/README.md'
        print(f'@@@@  model: {name}  @@@@')
        params = m['Args']
        save_path_model = f'{base_dir}/{name}/'
        params['path_save'] = f'{save_path_model}models_interval/'
        # check_file_func = m['check_file_func']
        # file_check = check_file_func(base_dir, name)
        # if not os.path.exists(file_check) or sim_setting['overwrite']:
        if not os.path.exists(readme_file_name) or overwrite:
            write_readme_file(m, readme_file_name)
            fitted_Model = fit_model(X_train, Y_train, X_test, Y_test,
                                     m['run_func'], m['predict_func'], make_pred, m['Args'])
            save_func = m['save_func']
            save_func(fitted_Model, base_dir, name, save_model = True)
        
        
        print("=============================")


def config_save(sim_file, change_dir_save = ''):
    sim_setting = load_sim_settings(sim_file)
    base_dir = sim_setting['base_dir']
    base_dir = f'{base_dir}{change_dir_save}/'
    sim_config = configparser.ConfigParser()
    sim_config.read(sim_file)
    for model_name in sim_config['models']:
        print(model_name)
        config_file = sim_config['models'][model_name]
        config_model = configparser.ConfigParser()
        config_model.read(config_file)
        save_name = config_get(config_model, 'main', 'method',config_file)
        os.makedirs(f'{base_dir}/{save_name}/', exist_ok= True)
        shutil.copy(config_file, f'{base_dir+ save_name }/{os.path.basename(config_file)}')
        shutil.copy(sim_file, f'{base_dir+ save_name }/{os.path.basename(sim_file)}')
        


def save_train_ids(sim_file, Y_train, model_number):
    sim_setting = load_sim_settings(sim_file)
    base_dir = sim_setting['base_dir']
    model_name = list((sim_setting['models']).keys())[0]
    os.makedirs(f'{base_dir}/{model_name}_{model_number}/', exist_ok= True)
    # Save the DataFrame index to a .npy file
    index_array = Y_train.index.to_numpy()
    np.save(f'{base_dir}/{model_name}_{model_number}/{model_name}_{model_number}_trainBins.npy',
            index_array, allow_pickle=True)


########################################################################################

def load_data_sim(sim_setting):
    
    path_X_test = sim_setting['path_X_test']
    path_X_train = sim_setting['path_X_train']
    path_Y_test = sim_setting['path_Y_test']
    path_Y_train = sim_setting['path_Y_train']
    scale = ast.literal_eval(sim_setting['scale'])
    
    remove_unMutated = ast.literal_eval(sim_setting['remove_unMutated'])
    
    X_train, Y_train, X_test, Y_test = create_TestTrain_TwoSources(path_X_train, 
                                                               path_Y_train, 
                                                               path_X_test, 
                                                               path_Y_test,
                                                               scale)
        
    
        
    if remove_unMutated:
        Y_train = Y_train[Y_train['nMut'] != 0]
        X_train = X_train.loc[Y_train.index]
        
        Y_test = Y_test[Y_test['nMut'] != 0]
        X_test = X_test.loc[Y_test.index]
    
    if (Y_test.index != X_test.index).all():
        raise ValueError('X_test and Y_test indexes are not the same')
    if (Y_train.index != X_train.index).all():
        raise ValueError('X_train and Y_train indexes are not the same')
        
    return X_train, Y_train, X_test, Y_test


