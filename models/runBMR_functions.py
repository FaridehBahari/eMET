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
# if platform.system() == 'Linux':
#     from pybedtools import BedTool
#     import pybedtools
    


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


def RUN_pairRank(sim_setting,  X_train, Y_train, X_test, Y_test, overwrite = True):
    
    models = sim_setting['models']
    base_dir = sim_setting['base_dir']
      
    for key in models:
       
        m = models[key]
        save_name = m['save_name']
        save_path_model = f'{base_dir}/{save_name}/'
        os.makedirs(f'{base_dir}/{save_name}/', exist_ok= True)
        readme_file_name = f'{base_dir}/{save_name}/README.md'
        print(f'@@@@  model: {save_name}  @@@@')
        if not os.path.exists(readme_file_name) or overwrite:
            write_readme_file(m, readme_file_name)
            
            run_func = m['run_func']
            params = m['Args']
            params['path_save'] = f'{save_path_model}models_interval/'
            model_data = run_func(X_train, Y_train, params)
            model = model_data['model']
            params = model_data['NN_hyperparams']
            
            model.save(f'{save_path_model+save_name}_model.h5')
            
            # Save the dictionary to a file using pickle
            with open(f'{base_dir}/{save_name}/{save_name}_params.pkl', 'wb') as f: 
                pickle.dump(params, f)
        
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


def sample_train_valvar(var_interval_response, Y_train, 
                               train_info, val_size, seed_value):
    
    # restrict to > 20nt length variable-size bins
    
    var_interval_response = var_interval_response.iloc[np.where(var_interval_response.length >= 20)]
    
    # sample from variable-size bins to have validation set
    np.random.seed(seed_value)
    val_indices = np.random.choice(var_interval_response.index, 
                                    size=val_size, replace=False)
    Y_val = var_interval_response.loc[val_indices]
    
    # annotate fixed-size bins
    train_Y_annotated = pd.concat([Y_train, train_info], axis=1)
    
    # remove validation bins from train set
    filtered_train_Y = train_Y_annotated[~train_Y_annotated['orig_name'].str.contains('|'.join(Y_val.index))]
    
    return filtered_train_Y, Y_val



def val_IDs_fixedElems(bed_tr, bed_val, seed_value, val_size):
    
    
    
    bed_val = bed_val.iloc[np.where(bed_val[2] - bed_val[1] >= 20)]
    
    np.random.seed(seed_value)
    tr_indices = np.random.choice(bed_tr.index, size=val_size, replace=False)
    bed_tr = bed_tr.loc[tr_indices]
    
    bedObj_val = BedTool.from_dataframe(bed_val)
    
    
    bedObj_tr = BedTool.from_dataframe(bed_tr)
    
    intersection_tr= bedObj_tr.intersect(bedObj_val).to_dataframe()
    non_train_set_binIDs = np.unique(intersection_tr.name)
    
    intersection_val = bedObj_val.intersect(bedObj_tr).to_dataframe()
    val_set_binIDs = np.unique(intersection_val.name)
    
    
    return val_set_binIDs, non_train_set_binIDs


def sample_train_val_fixedSize(Y_train, Y_val, bed_tr, bed_var, seed_value, val_size):
    
    val_set_binIDs, non_train_set_binIDs = val_IDs_fixedElems(bed_tr, bed_var, seed_value, val_size)
    Y_train = Y_train[~Y_train.index.isin(non_train_set_binIDs) ]
    Y_val = Y_val.loc[val_set_binIDs]
    
    return Y_train, Y_val
    


def select_groups_from_dict(dictionary, keys_to_include):
    
    # Create an empty list to store the values
    included_values = []

    # Iterate through the original dictionary
    for key, value in dictionary.items():
        # Check if the key should be included
        if key in keys_to_include:
            # Extend the list with the values
            included_values.extend(value)

    return included_values


def get_features_category(category, path_featureURLs = '../external/database/all_feature_URLs.xlsx'):
    
    # # Load feature groups from Excel file
    # feature_groups_df = pd.read_excel(path_featureURLs)  
    # feature_groups = feature_groups_df.groupby('Group Name')['Feature Name'].apply(list).to_dict()
    # nucleotide_content = ['ACA', 'ACC', 'ACG', 'ACT', 'ATA', 'ATC', 'ATG', 'ATT',
    #                       'CCA', 'CCC', 'CCG', 'CCT', 'CTA', 'CTC', 'CTG', 'CTT', 
    #                       'GCA', 'GCC', 'GCG', 'GCT', 'GTA', 'GTC', 'GTG', 'GTT', 
    #                       'TCA', 'TCC', 'TCG', 'TCT', 'TTA', 'TTC', 'TTG', 'TTT', 
    #                       'TA5p', 'TC5p', 'TG5p', 'TT5p', 'CA5p', 'CC5p', 'CG5p', 
    #                       'CT5p', 'AT3p', 'CT3p', 'GT3p', 'AC3p', 'GC3p', 'TC3p']
    
    
    # # Adding 'nucleotide content'and 'APOBEC' key to the dictionary
    # feature_groups['nucleotide content'] = nucleotide_content
    # feature_groups['APOBEC'] = ['APOBEC3A']
    
    
    
    # # # File path from where to load the dictionary
    # # file_path = '../external/procInput/ftrs_dict.pickle'

    # # # Save the dictionary to disk
    # # with open(file_path, 'wb') as file:
    # #     pickle.dump(feature_groups, file)
    
    
    # Load the dictionary from disk
    with open('../external/procInput/ftrs_dict.pickle', 'rb') as file:
        feature_groups = pickle.load(file)

    
    features = select_groups_from_dict(feature_groups, category)
    
    return features




def load_data_sim(sim_setting, category = ['DNA_accessibility', 'Epigenetic_mark', 'HiC', 
                    'RNA_expression', 'Replication_timing', 'conservation',
                    'nucleotide content']):
    
    path_X_test = sim_setting['path_X_test']
    path_X_train = sim_setting['path_X_train']
    path_Y_test = sim_setting['path_Y_test']
    path_Y_train = sim_setting['path_Y_train']
    scale = ast.literal_eval(sim_setting['scale'])
    DSmpl = ast.literal_eval(sim_setting['DSmpl'])
    n_sample = sim_setting['n_sample']
    remove_unMutated = ast.literal_eval(sim_setting['remove_unMutated'])
    
    if len(category) != 0:
        ftrs = get_features_category(category)
        
        X_train, Y_train, X_test, Y_test = create_TestTrain_TwoSources(path_X_train, 
                                                                   path_Y_train, 
                                                                   path_X_test, 
                                                                   path_Y_test,
                                                                   scale, use_features = ftrs)
        X_train = X_train.loc[:, ftrs]
        X_test = X_test.loc[:, ftrs]
    
    else:
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
    
    if DSmpl:
        
        np.random.seed(40)
        tr_indices = np.random.choice(list(Y_train.index), size=n_sample, replace=False)
        Y_train = Y_train.loc[tr_indices]
        print(f'Down sampling was performed... number of training bins: {Y_train.shape[0]}')
        X_train = X_train.loc[Y_train.index]
    
    if (Y_test.index != X_test.index).all():
        raise ValueError('X_test and Y_test indexes are not the same')
    if (Y_train.index != X_train.index).all():
        raise ValueError('X_train and Y_train indexes are not the same')
        
    return X_train, Y_train, X_test, Y_test


def load_data_sim_2(sim_setting, category = ['DNA_accessibility', 'Epigenetic_mark', 'HiC', 
                    'RNA_expression', 'Replication_timing', 'conservation',
                    'nucleotide content']):
    
    path_X_train = sim_setting['path_X_train']
    path_X_val = sim_setting['path_X_validate']
    
    
    
    path_Y_train = sim_setting['path_Y_train']
    path_Y_val = sim_setting['path_Y_validate']
    
    scale = ast.literal_eval(sim_setting['scale'])
    
    DSmpl = ast.literal_eval(sim_setting['DSmpl'])
    n_sample = sim_setting['n_sample']
    remove_unMutated = ast.literal_eval(sim_setting['remove_unMutated'])
    
    if len(category) != 0:
        # load all train 
        ftrs = get_features_category(category)
        X_tr_cmplt, Y_tr_cmplt = load_data(path_X_train, path_Y_train,
                                           use_features=ftrs)
        
        
        # load all val 
        X_val_cmplt, Y_val_cmplt = load_data(path_X_val, path_Y_val,
                                             use_features=ftrs)
    else:
        # load all train 
        X_tr_cmplt, Y_tr_cmplt = load_data(path_X_train, path_Y_train)        
        
        # load all val 
        X_val_cmplt, Y_val_cmplt = load_data(path_X_val, path_Y_val)
       
    if scale:
        X_tr_cmplt, meanSc, sdSc = scale_train(X_tr_cmplt)
        X_val_cmplt = scale_test(X_val_cmplt, meanSc, sdSc)
    
    
    
    if remove_unMutated:
        Y_tr_cmplt = Y_tr_cmplt[Y_tr_cmplt['nMut'] != 0]
        X_tr_cmplt = X_tr_cmplt.loc[Y_tr_cmplt.index]
        
        Y_val_cmplt = Y_val_cmplt[Y_val_cmplt['nMut'] != 0]
        X_val_cmplt = X_val_cmplt.loc[Y_val_cmplt.index]
    
    
    if DSmpl:
        np.random.seed(0)
        tr_indices = np.random.choice(list(Y_tr_cmplt.index), size=n_sample, replace=False)
        Y_tr_cmplt = Y_tr_cmplt.loc[tr_indices]
        print(f'Down sampling was performed... number of training bins: {Y_tr_cmplt.shape[0]}')
        X_tr_cmplt = X_tr_cmplt.loc[Y_tr_cmplt.index]
    
    
    if (Y_val_cmplt.index != X_val_cmplt.index).all():
        raise ValueError('X_val and Y_val indexes are not the same')
    if (X_tr_cmplt.index != Y_tr_cmplt.index).all():
        raise ValueError('X_train and Y_train indexes are not the same')
        
    return X_tr_cmplt, Y_tr_cmplt, X_val_cmplt, Y_val_cmplt

###############################################################################
# this section was used for assessing mutation status and length it can be replaced with the previous ones if the results are the same

def sample_validations(Y_train, Y_val, val_size, seed_value):
    
    Y_val = Y_val.iloc[np.where(Y_val.length >= 20)]
    Y_val = Y_val.iloc[np.where(Y_val.nMut != 0)]
    
    # sample from variable-size bins to have validation set
    np.random.seed(seed_value)
    val_indices = np.random.choice(Y_val.index, 
                                    size=val_size, replace=False)
    Y_val = Y_val.loc[val_indices]
    
    # remove validation bins from train set
    filtered_train_Y = Y_train.loc[~Y_train.index.isin(val_indices)]
    
    return filtered_train_Y, Y_val



def repeated_train_test(sim_setting,  X_tr_cmplt, Y_tr_cmplt, X_val_cmplt, Y_val_cmplt,
            make_pred = True, overwrite = True, n_repeat = 10, length_filter = None):
    
    if(length_filter):
        
        Y_tr_cmplt = Y_tr_cmplt.iloc[np.where(Y_tr_cmplt.length > length_filter)]
        X_tr_cmplt = X_tr_cmplt.loc[Y_tr_cmplt.index]
    
    fixed_size_train = ast.literal_eval(sim_setting['fixed_size_train'])
    path_train_info = sim_setting['path_train_info']
    path_bed_tr = sim_setting['path_bed_tr']
    path_bed_var = sim_setting['path_bed_var']
    
    models = sim_setting['models']
    base_dir = sim_setting['base_dir']
    
    val_size = X_tr_cmplt.shape[0]//5 
    Nr_pair_acc = sim_setting['Nr_pair_acc']
    
    if path_train_info != '':
        '----- using trainInfo ----'
        train_info = pd.read_csv(path_train_info, sep = '\t', index_col='binID')
        train_info = train_info.loc[Y_tr_cmplt.index]
    elif fixed_size_train:
        '------ using pybedtools -----'
        bed_tr = pd.read_csv(path_bed_tr, sep = '\t', header = None)
        bed_tr['binID'] = bed_tr[3]
        bed_tr = bed_tr.set_index('binID')
        bed_tr = bed_tr.loc[Y_tr_cmplt.index]
        
        bed_val = pd.read_csv(path_bed_var, sep = '\t', header = None)
        bed_val['binID'] = bed_val[3]
        bed_val = bed_val.set_index('binID')
        bed_val = bed_val.loc[Y_val_cmplt.index]
        
    seed_values = [1, 5, 14, 10, 20, 30, 40, 50, 60, 70, 80, 90, 77, 100, 110]
    
    
    for key in models:
       
        m = models[key]
        name = m['save_name']
        
        os.makedirs(f'{base_dir}/{name}/', exist_ok= True)
        readme_file_name = f'{base_dir}/{name}/README.md'
        print(f'@@@@  model: {name}  @@@@')
        params = m['Args']
        save_path_model = f'{base_dir}/{name}/'
        
        # check_file_func = m['check_file_func']
        # file_check = check_file_func(base_dir, name)
        # if not os.path.exists(file_check) or sim_setting['overwrite']:
        if not os.path.exists(readme_file_name) or overwrite:
            write_readme_file(m, readme_file_name)
            
            for i in range(n_repeat):
                params['path_save'] = f'{save_path_model}rep_train_test/models{i+1}/'
                print(f'.......... repeat number {i+1} of train-test for evaluation of the {name} ......')
                seed_value = np.random.seed(seed_values[i])
                
                if os.path.exists(f'{save_path_model}/rep_train_test/{name}_M{i+1}_assessment.tsv'):
                    print(f"Skipping iteration {i+1} as the file already exists.")
                    continue
                
                if path_train_info != '':
                    '----- using trainInfo ----'
                    Y_train, Y_test = sample_train_valvar(Y_val_cmplt, Y_tr_cmplt, 
                                                train_info, val_size, seed_value)
                elif fixed_size_train: 
                    '------ using pybedtools -----'
                    Y_train, Y_test = sample_train_val_fixedSize(Y_tr_cmplt, Y_val_cmplt, bed_tr,
                                                                  bed_val, seed_value, val_size)
                    
                else: 
                    Y_train, Y_test = sample_validations(Y_tr_cmplt, Y_val_cmplt, 
                                                          val_size, seed_value)
                    
                X_train = X_tr_cmplt.loc[Y_train.index]
                X_test = X_val_cmplt.loc[Y_test.index]
                
                print(X_train.shape)
                print(X_test.shape)
                
                common_indices = X_test.index.intersection(X_train.index)
                
                if not common_indices.empty:
                    raise ValueError(f"Common indices found between X_test and X_train:{common_indices}")
                else:
                    print("No common indices found between X_test and X_train.")
                    
                fitted_Model = fit_model(X_train, Y_train, X_test, Y_test,
                                         m['run_func'], m['predict_func'], make_pred, m['Args'])
                save_func = m['save_func']
                itr = i+1
                                
                save_func(fitted_Model, base_dir, name, iteration=itr, save_model=True)
                
                Y_pred = fitted_Model.predRates_test
                Y_obs = Y_test.nMut/(Y_test.N * Y_test.length)
                assessments = assess_model(Y_pred, Y_obs, Nr_pair_acc, name, per_element=False)
                
                path_assessments = f'{save_path_model}/rep_train_test/{name}_M{i+1}_assessment.tsv'
                assessments.to_csv(path_assessments, sep='\t')
        
        print("=============================")
        dir_path = f'{save_path_model}/rep_train_test/'
        save_metrics_summary(dir_path)
        


#########################################################################
# the following functions were used for running repeated train and test on fixed-size elements 
# for NNs. where pybedtools was not installed on cancer_gpu and the pyb env was not able to handle 
# GPU memory

def extract_bin_size(path_bed_tr):
    
    if path_bed_tr == '../external/database/bins/proccessed/intergenic_fixed1M.bed6':
        bin_size = '1M'
    elif path_bed_tr == '../external/database/bins/proccessed/intergenic_fixed100k.bed6':
        bin_size = '100k'
    elif path_bed_tr == '../external/database/bins/proccessed/intergenic_fixed50k.bed6':
        bin_size = '50k'
    elif path_bed_tr == '../external/database/bins/proccessed/intergenic_fixed10k.bed6':
        bin_size = '10k'
        
    return bin_size


def sample_train_val_fixedSize2(Y_train, Y_val, bed_tr, bed_var, 
                               path_bed_tr, iteration):
    
    bin_size = extract_bin_size(path_bed_tr)
    saved_file = f'../external/BMR/procInput/fixedSize_trainValIDs/fixed{bin_size}_IDs.pkl'
    
    with open(saved_file, 'rb') as f:
         results = pickle.load(f)
    
    val_set_binIDs = results[f'{bin_size}_{iteration}']['val_set_binIDs']
    non_train_set_binIDs = results[f'{bin_size}_{iteration}']['non_train_set_binIDs']
    
    Y_train = Y_train[~Y_train.index.isin(non_train_set_binIDs) ]
    Y_val = Y_val.loc[val_set_binIDs]
    
    return Y_train, Y_val



def repeated_train_test2(sim_setting,  X_tr_cmplt, Y_tr_cmplt, X_val_cmplt, Y_val_cmplt,
            make_pred = True, overwrite = True, n_repeat = 10):
    
    
    path_bed_tr = sim_setting['path_bed_tr']
    path_bed_var = sim_setting['path_bed_var']
    
    models = sim_setting['models']
    base_dir = sim_setting['base_dir']
    
    Nr_pair_acc = sim_setting['Nr_pair_acc']
    
    bed_tr = pd.read_csv(path_bed_tr, sep = '\t', header = None)
    bed_tr['binID'] = bed_tr[3]
    bed_tr = bed_tr.set_index('binID')
    bed_tr = bed_tr.loc[Y_tr_cmplt.index]
    
    bed_val = pd.read_csv(path_bed_var, sep = '\t', header = None)
    bed_val['binID'] = bed_val[3]
    bed_val = bed_val.set_index('binID')
    bed_val = bed_val.loc[Y_val_cmplt.index]
     
    for key in models:
       
        m = models[key]
        name = m['save_name']
        
        os.makedirs(f'{base_dir}/{name}/', exist_ok= True)
        readme_file_name = f'{base_dir}/{name}/README.md'
        print(f'@@@@  model: {name}  @@@@')
        params = m['Args']
        save_path_model = f'{base_dir}/{name}/'
        
        # check_file_func = m['check_file_func']
        # file_check = check_file_func(base_dir, name)
        # if not os.path.exists(file_check) or sim_setting['overwrite']:
        if not os.path.exists(readme_file_name) or overwrite:
            write_readme_file(m, readme_file_name)
            
            for i in range(n_repeat):
                params['path_save'] = f'{save_path_model}rep_train_test/models{i+1}/'
                print(f'.......... repeat number {i+1} of train-test for evaluation of the {name} ......')
                                
                if os.path.exists(f'{save_path_model}/rep_train_test/{name}_M{i+1}_assessment.tsv'):
                    print(f"Skipping iteration {i+1} as the file already exists.")
                    continue
                
                Y_train, Y_test = sample_train_val_fixedSize2(Y_tr_cmplt, Y_val_cmplt,
                                                              bed_tr, bed_val,
                                                              path_bed_tr, i)
                
                print('===========')
                
                X_train = X_tr_cmplt.loc[Y_train.index]
                X_test = X_val_cmplt.loc[Y_test.index]
                
                print(X_train.shape)
                print(X_test.shape)
                
                common_indices = X_test.index.intersection(X_train.index)
                
                if not common_indices.empty:
                    raise ValueError(f"Common indices found between X_test and X_train:{common_indices}")
                else:
                    print("No common indices found between X_test and X_train.")
                    
                fitted_Model = fit_model(X_train, Y_train, X_test, Y_test,
                                         m['run_func'], m['predict_func'], make_pred, m['Args'])
                save_func = m['save_func']
                itr = i+1
                                
                save_func(fitted_Model, base_dir, name, iteration=itr, save_model=True)
                
                Y_pred = fitted_Model.predRates_test
                Y_obs = Y_test.nMut/(Y_test.N * Y_test.length)
                assessments = assess_model(Y_pred, Y_obs, Nr_pair_acc, name, per_element=False)
                
                path_assessments = f'{save_path_model}/rep_train_test/{name}_M{i+1}_assessment.tsv'
                assessments.to_csv(path_assessments, sep='\t')
        
        print("=============================")
        dir_path = f'{save_path_model}/rep_train_test/'
        save_metrics_summary(dir_path)

