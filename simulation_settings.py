############# TODO ____ fullset number of epoches should be set ###########
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 16:13:00 2023

@author: Farideh
"""
import sys
from models.GBM_functions import gbm_model_info


import configparser

def model_info_load(model_info_func_name, model_file, config_file):
    model_info_func = globals()[model_info_func_name]
    model = model_info_func(model_file, config_file)
    return model

def config_get(config, category, item, sim_file):
    try:
        ret = config.get(category, item)
    except Exception as e:
        print(f"Error: ({category}, {item}) does not exist in {sim_file}")
        sys.exit(0)
        return None
    return ret

    
def load_sim_settings(sim_file):
    sim_config = configparser.ConfigParser()
    sim_config.read(sim_file)
    
    models = {}
    for model_name in sim_config['models']:
        print(model_name)
        config_file = sim_config['models'][model_name]
        config_model = configparser.ConfigParser()
        config_model.read(config_file)
        save_name = config_get(config_model, 'main', 'method',config_file)
        
        model_info_func_str = config_get(config_model, 'main', 'model_info_func',config_file)
        models[save_name] = model_info_load(model_info_func_str, save_name, 
                                            config_file)
    
    
    settings = {
        'path_X_test': config_get(sim_config, 'main', 'path_X_test',sim_file),
        'path_X_train': config_get(sim_config, 'main', 'path_X_train',sim_file),
        'path_Y_train': config_get(sim_config, 'main', 'path_Y_train',sim_file),
        'path_Y_test': config_get(sim_config, 'main', 'path_Y_test',sim_file),
        'scale': config_get(sim_config, 'main', 'scale',sim_file),
        'models': models,
        'base_dir': config_get(sim_config, 'main', 'base_dir',sim_file),
        'remove_unMutated': config_get(sim_config, 'main', 'remove_unMutated',sim_file),
        'Nr_pair_acc': int(config_get(sim_config, 'main', 'Nr_pair_acc',sim_file))
    }
    return settings


def load_sim_settings_perBatchPerf(dir_path, setting_config, param_config):
    sim_file = f'{dir_path+setting_config}'
    sim_config = configparser.ConfigParser()
    sim_config.read(sim_file)
    
    models = {}
    for model_name in sim_config['models']:
        print(model_name)
        config_file = f'{dir_path+param_config}'
        config_model = configparser.ConfigParser()
        config_model.read(config_file)
        save_name = config_get(config_model, 'main', 'method',config_file)
        
        model_info_func_str = config_get(config_model, 'main', 'model_info_func',config_file)
        models[save_name] = model_info_load(model_info_func_str, save_name, 
                                            config_file)
    
    
    settings = {
        'path_X_test': config_get(sim_config, 'main', 'path_X_test',sim_file),
        'path_X_train': config_get(sim_config, 'main', 'path_X_train',sim_file),
        'path_Y_train': config_get(sim_config, 'main', 'path_Y_train',sim_file),
        'path_Y_test': config_get(sim_config, 'main', 'path_Y_test',sim_file),
        'scale': config_get(sim_config, 'main', 'scale',sim_file),
        'models': models,
        'base_dir': config_get(sim_config, 'main', 'base_dir',sim_file),
        'remove_unMutated': config_get(sim_config, 'main', 'remove_unMutated',sim_file),
        'Nr_pair_acc': int(config_get(sim_config, 'main', 'Nr_pair_acc',sim_file))
        
    }
    return settings