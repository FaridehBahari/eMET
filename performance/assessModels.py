# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 16:13:00 2023

@author: Farideh
"""
import pandas as pd
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append('models/')
from readFtrs_Rspns import read_response
from scipy.stats import spearmanr
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


from tensorflow.keras.models import load_model

import warnings
def read_pred(path_pred):
    Y_pred = pd.read_csv(path_pred, sep = "\t", header=0, index_col='binID',
                         usecols=['binID', 'predRate'])
    # Load the CSV file into a DataFrame
    df = pd.read_csv('../external/BMR/procInput/ann_PCAWG_ID_complement.csv', sep=',')

    filtered_df = df[(df['in_CGC'] | df['in_CGC_literature'] | df['in_CGC_new'] | df['in_oncoKB'] | df['in_pcawg'])]
        
    # Select the 'PCAWG_IDs' column from the filtered DataFrame
    drivers = filtered_df['PCAWG_IDs']
    Y = Y_pred.loc[~(Y_pred.index).isin(drivers)]
    
    return Y

def read_obs(path_Y, remove_unMut):
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv('../external/BMR/procInput/ann_PCAWG_ID_complement.csv', sep=',')

    filtered_df = df[(df['in_CGC'] | df['in_CGC_literature'] | df['in_CGC_new'] | df['in_oncoKB'] | df['in_pcawg'])]
        
    # Select the 'PCAWG_IDs' column from the filtered DataFrame
    drivers = filtered_df['PCAWG_IDs']

    Y_all = read_response(path_Y)
    Y = Y_all.loc[~(Y_all.index).isin(drivers)]
    
    if remove_unMut:
        Y = Y[Y['nMut'] != 0]
    Y_obs = Y['obsRates']
    Y_obs = Y_obs.to_frame(name = 'obsRates')
    return Y_obs

def split_by_element(df, element_type):
    df_element = df.loc[df.index.str.contains(element_type)]
    return df_element


def calc_corr(pred, obs):
    corr, p_value = spearmanr(pred, obs)
    
    return corr


def generate_pair_indices(N_pairs, N_tot):
    
    # retain uniqe rows
    pairs = np.unique(np.column_stack((np.random.choice(N_tot, N_pairs, replace=True),
                                       np.random.choice(N_tot, N_pairs, replace=True))), axis=0)
    
    # just retain unique pairs per row (pair i,i is not alloweded)
    non_duplicate_rows = pairs[~np.all(pairs[:, 1:] == pairs[:, :-1], axis=1)]
    
    return non_duplicate_rows 

def calc_Pairs_acc(Nr_pair_acc, obs, pred):
    
    pairs = generate_pair_indices(Nr_pair_acc, obs.shape[0])
    pred_res = pred.iloc[pairs[:, 0]].values > pred.iloc[pairs[:, 1]].values
    obs_res = obs.iloc[pairs[:, 0]].values > obs.iloc[pairs[:, 1]].values
    
    return np.mean(pred_res == obs_res)


def assess_model_element_type(Y_pred, Y_obs, Nr_pair_acc, model_name, elem):
    
    if elem == "intergenic":
        obs_elem = split_by_element(Y_obs,  r'[v]\d')
        pred_elems = split_by_element(Y_pred,  r'[v]\d')
    else:
        obs_elem = split_by_element(Y_obs, elem)
        pred_elems = split_by_element(Y_pred, elem)
    
    pred_elem = pred_elems[~np.isinf(pred_elems)]
    pred_elem = pred_elem[~np.isnan(pred_elem)]
    if pred_elem.shape[0] != pred_elems.shape[0]:
        warnings.warn(f"{elem} prediction contains NaN or infinite values. These elements will be discarded.", stacklevel=1)
    obs_elem = obs_elem.loc[pred_elem.index]
    
    corr_elem =  calc_corr(pred_elem, obs_elem)
    if Nr_pair_acc != 0:
        acc_elem = calc_Pairs_acc(Nr_pair_acc, obs_elem, pred_elem)
    else:
        acc_elem = np.nan
    mse_elem = mean_squared_error(obs_elem, pred_elem)
    mae_elem = mean_absolute_error(obs_elem, pred_elem)
    
    return corr_elem, acc_elem, mse_elem, mae_elem
    

def assess_model(Y_pred, Y_obs, Nr_pair_acc, model_name, per_element=True):
    
    acc_name = f"acc_{model_name}"
    mse_name = f"mse_{model_name}"
    corr_name  = f"corr_{model_name}"
    mae_name = f"made_{model_name}"
    
    if per_element:
        elems = ["gc19_pc.cds", "enhancers", "gc19_pc.3utr", "gc19_pc.5utr",
                 "gc19_pc.promCore", "gc19_pc.ss"] #, "lncrna.ncrna",  "lncrna.promCore"
        if sum(Y_pred.index.str.contains(r'[v]\d')) != 0:
            elems.append("intergenic")
        
        results = []
        for elem in elems:
            corr_elem, acc_elem, mse_elem, mae_elem = assess_model_element_type(Y_pred, 
                                                                      Y_obs, Nr_pair_acc,
                                                                      model_name,
                                                                      elem)
            
            results.append({'Element': elem, acc_name : acc_elem,
                            corr_name : corr_elem, 
                            mse_name : mse_elem,
                            mae_name: mae_elem})
            
        performances = pd.DataFrame(results).set_index('Element').pivot_table(index=None, columns='Element')
        
    else:
        Y_obs = Y_obs.loc[Y_pred.index]
        corr =  calc_corr(Y_pred, Y_obs)
        
        if Nr_pair_acc != 0:
            acc = calc_Pairs_acc(Nr_pair_acc, Y_obs, Y_pred)
        else:
            acc = np.nan
        
        mse = mean_squared_error(Y_obs, Y_pred)
        mae = mean_absolute_error(Y_obs, Y_pred)
        
        results = {'Element': 'train', acc_name: [acc],
                   corr_name: [corr], mse_name: [mse],
                   mae_name: [mae]}
        performances = pd.DataFrame(results).set_index('Element').pivot_table(index=None, columns='Element')
    
    return performances


def assess_models(sim_setting):
    models = sim_setting['models']
    path_Y_train = sim_setting['path_Y_train']
    path_Y_test = sim_setting['path_Y_test']
    
    base_dir = sim_setting['base_dir']
    Nr_pair_acc = sim_setting['Nr_pair_acc']
    
    # remove_unMutated = ast.literal_eval(sim_setting['remove_unMutated'])
    remove_unMutated = True
    
    Y_obs_all_intergenic = read_obs(path_Y_train, remove_unMutated)
    Y_obs_all_elems = read_obs(path_Y_test, remove_unMutated)
    
    # acc_all = []
    # corr_all = []
    # mse_all = []
    
    for key in models:
        m = models[key]
        # load train
        save_name = m['save_name']
        print(save_name)
        
        Y_obs_unseen = Y_obs_all_elems.copy()
        Y_obs_seen = Y_obs_all_intergenic.copy()
            
        path_pred_unseen = f'{base_dir}/{save_name}/{save_name}_predTest.tsv'
        Y_pred_unseen = read_pred(path_pred_unseen)
        if Y_pred_unseen.shape[0] <= Y_obs_unseen.shape[0]:
            Y_obs_unseen = Y_obs_unseen.loc[Y_pred_unseen.index]
        elif Y_pred_unseen.shape[0] > Y_obs_unseen.shape[0]:
            Y_pred_unseen = Y_pred_unseen.loc[Y_obs_unseen.index]
        
        if (Y_pred_unseen.index != Y_obs_unseen.index).all():
            raise ValueError('index mismatch')
        assessments_test = assess_model(Y_pred_unseen, Y_obs_unseen, 
                                        Nr_pair_acc, save_name, per_element=True)
        
        
        
        path_pred_seen = f'{base_dir}/{save_name}/{save_name}_predTrain.tsv'
        Y_pred_seen = read_pred(path_pred_seen)
        
        if Y_pred_seen.shape[0] <= Y_obs_seen.shape[0]:
            Y_obs_seen = Y_obs_seen.loc[Y_pred_seen.index]
        elif Y_pred_seen.shape[0] > Y_obs_seen.shape[0]:
            Y_pred_seen = Y_pred_seen.loc[Y_obs_seen.index]
        
        if (Y_pred_seen.index != Y_obs_seen.index).all():
            ValueError('index mismatch')
         
         
        assessments_train = assess_model(Y_pred_seen, Y_obs_seen, Nr_pair_acc,
                                         save_name, per_element=False)
        
        assessments = pd.concat([assessments_test, assessments_train], axis=1)
        
        assessments.to_csv(f'{base_dir}/{save_name}/{save_name}_assessments.tsv', sep='\t')
        
        # acc_all.append(assessments.loc['acc_'+save_name])
        # corr_all.append(assessments.loc['corr_'+save_name])
        # mse_all.append(assessments.loc['mse_'+save_name])
        
        print("=========================")
    
    # acc_all_df = pd.concat(acc_all, axis=1)
    # corr_all_df = pd.concat(corr_all, axis=1)
    # mse_all_df = pd.concat(mse_all, axis=1)
    
    # acc_all_df.to_csv(f'{base_dir}/acc_all.tsv', sep='\t')
    # corr_all_df.to_csv(f'{base_dir}/corr_all.tsv', sep='\t')
    # mse_all_df.to_csv(f'{base_dir}/mse_all.tsv', sep='\t')
    

        
###################################################################3
def load_all_obsRates():
    path_Y_test = '../external/rawInput/Pan_Cancer_test_y.tsv'
    path_Y_train = '../external/rawInput/Pan_Cancer_train_y.tsv'
    
    Y_test = read_response(path_Y_test) 
    Y_train = read_response(path_Y_train) 
    Y_all = pd.concat([Y_test, Y_train], axis=0)
    return Y_all


def extract_Y_obs(path_preds):
    Yobs_all = load_all_obsRates()
    Y_pred = read_pred(path_preds)
    Y_obs = Yobs_all.loc[Y_pred.index]
    Y_obs = Y_obs[Y_obs.nMut != 0]
    Y_obs = pd.DataFrame(Y_obs.obsRates)
    return Y_obs
 

    


def assess_models_new(sim_setting):
    models = sim_setting['models']
    base_dir = sim_setting['base_dir']
    Nr_pair_acc = sim_setting['Nr_pair_acc']
    # model_name = list(models.keys())[0]
    # save_name = sim_setting['models'][model_name]['save_name']
    
    for key in models:
       
        m = models[key]
        save_name = m['save_name']   
        
        print(f'assessment for {save_name}')
        path_predTest = f'{base_dir}/{save_name}/{save_name}_predTest.tsv'

        Y_pred_test = read_pred(path_predTest)
        Y_obs_test = extract_Y_obs(path_predTest)

        path_predTrain = f'{base_dir}/{save_name}/{save_name}_predTrain.tsv'
        Y_pred_train = read_pred(path_predTrain)
        Y_obs_train = extract_Y_obs(path_predTrain)


        test_ensemble = assess_model(Y_pred_test, Y_obs_test, Nr_pair_acc = Nr_pair_acc, 
                     model_name = save_name, per_element = True)

        train_ensemble = assess_model(Y_pred_train, Y_obs_train, Nr_pair_acc = Nr_pair_acc, 
                     model_name = save_name, per_element = False)

        assessments = pd.concat([test_ensemble, train_ensemble], axis=1)

        assessments.to_csv(f'{base_dir}/{save_name}/{save_name}_assessments.tsv', sep='\t')
    
    


def assess_model_number_n(sim_setting, base_dir, save_name):
    Nr_pair_acc = sim_setting['Nr_pair_acc']
    
    path_predTest = f'{base_dir}/{save_name}/{save_name}_predTest.tsv'

    Y_pred_test = read_pred(path_predTest)
    Y_obs_test = extract_Y_obs(path_predTest)

    path_predTrain = f'{base_dir}/{save_name}/{save_name}_predTrain.tsv'
    Y_pred_train = read_pred(path_predTrain)
    Y_obs_train = extract_Y_obs(path_predTrain)


    test_ensemble = assess_model(Y_pred_test, Y_obs_test, Nr_pair_acc = Nr_pair_acc, 
                 model_name = save_name, per_element = True)

    train_ensemble = assess_model(Y_pred_train, Y_obs_train, Nr_pair_acc = Nr_pair_acc, 
                 model_name = save_name, per_element = False)

    assessments = pd.concat([test_ensemble, train_ensemble], axis=1)

    assessments.to_csv(f'{base_dir}/{save_name}/{save_name}_assessments.tsv', sep='\t')
