import pickle
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from simulation_settings import load_sim_settings
from readFtrs_Rspns import split_by_element, load_regulatory_elems
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from performance.assessModels import assess_model
from sklearn.metrics import mean_squared_error
from models.GBM_functions import run_gbm, predict_gbm
from models.runBMR_functions import  get_features_category

def generate_nonDriver_Dat(X_regLmnt, Y_regLmnt, drivers):
    
    # subset drivers to drivers of nMut>0 for our pan-cancer data or to element-specific drivers
    drivers = drivers[drivers.isin(Y_regLmnt.index)] 
    
    
    # Y_drivers = Y_regLmnt.loc[drivers]
    # X_drivers = X_regLmnt.loc[Y_drivers.index]
    
    Y_nonDrivers = Y_regLmnt.loc[~(Y_regLmnt.index).isin(drivers)]
    X_nonDrivers = X_regLmnt.loc[Y_nonDrivers.index]
    
    return X_nonDrivers, Y_nonDrivers #, X_drivers, Y_drivers

def bootstrap_samples(X_nonDrivers, Y_nonDrivers):
    n_samples = Y_nonDrivers.shape[0]
    indices = np.random.choice(X_nonDrivers.index.unique(), size=n_samples, replace=True)
    X_samples = X_nonDrivers.loc[indices]
    y_samples = Y_nonDrivers.loc[indices]
    return X_samples, y_samples, np.unique(indices)

def get_identified_drivers(path_ann_pcawg_IDs, based_on):
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(path_ann_pcawg_IDs, sep=',')
    
    if based_on == 'all':
        # Filter rows where at least one of the specified columns is TRUE
        filtered_df = df[(df['in_CGC'] | df['in_CGC_literature'] | df['in_CGC_new'] | df['in_oncoKB'] | df['in_pcawg'])] 
        
    # frequency_table = filtered_df['type_of_element'].value_counts()
    if based_on == 'in_pcawg':
        filtered_df = df[(df['in_pcawg'])]
        # frequency_table = filtered_df['type_of_element'].value_counts()
        
    # Select the 'PCAWG_IDs' column from the filtered DataFrame
    drivers = filtered_df['PCAWG_IDs']
    
    
    return drivers



def generate_test_train_bootstrapSamples(X_elem, Y_elem, drivers):
    
    X_nonDrivers, Y_nonDrivers = generate_nonDriver_Dat(X_elem, Y_elem, drivers)
    X_samples_train, y_samples_train, seen_bins = bootstrap_samples(X_nonDrivers, Y_nonDrivers)
    
    unseen_bins = Y_elem.index[np.where(~(Y_elem.index).isin(np.unique(seen_bins)))]
    y_samples_test = Y_elem.loc[np.unique(unseen_bins)]
    X_samples_test = X_elem.loc[np.unique(unseen_bins)]
    
    # check the number of bins in test and train data
    if len(np.unique(y_samples_train.index)) + len(np.unique(y_samples_test.index)) != Y_elem.shape[0]:
        raise ValueError('number of sample test_train indices is incompatible with the input data indices ')
        
    return X_samples_train, y_samples_train, X_samples_test, y_samples_test



def run_gbm_transferLearning(loaded_model, X_regLmnt, Y_regLmnt, param):
    
    # Reorder the columns of X_regLmnt according to the order in the loaded model
    model_ftr_names = loaded_model.feature_names  # Get feature names from the loaded model
    
    

    X_regLmnt, X_valid, Y_regLmnt, Y_valid = train_test_split(X_regLmnt, Y_regLmnt,
                                                        test_size=0.12, 
                                                        shuffle=True)
    # Ensure that the columns of X_regLmnt match the order of features in the loaded model
    X_regLmnt = X_regLmnt[model_ftr_names]
    X_valid = X_valid[model_ftr_names]
    
    if ((X_regLmnt.index != Y_regLmnt.index).all()):
            ValueError("The index values of X doesnt match the index values of Y.")
    
    # calculate base margin
    offset_train = np.array(np.log(Y_regLmnt.length+1/Y_regLmnt.N) + np.log(Y_regLmnt.N))
    offset_valid = np.array(np.log(Y_valid.length+1/Y_valid.N) + np.log(Y_valid.N))
    
    ftr_names = X_regLmnt.columns.values
    
    # dtrain = xgb.DMatrix(data=X_regLmnt, label=Y_regLmnt.nMut.values, feature_names=ftr_names)
    # dvalid = xgb.DMatrix(data=X_valid, label=Y_valid.nMut.values, feature_names=ftr_names)
    
    dtrain = xgb.DMatrix(data=X_regLmnt, label=Y_regLmnt.nMut.values, feature_names=ftr_names.tolist())
    dvalid = xgb.DMatrix(data=X_valid, label=Y_valid.nMut.values, feature_names=ftr_names.tolist())
    
    # add offset
    dtrain.set_base_margin(offset_train)
    dvalid.set_base_margin(offset_valid)
    
    # extract num_boost_round, early_stopping_rounds, and verbose_eval from param
    n_round = param['num_iteration']  #param.get('num_boost_round', 5000)
    # param.pop("num_iteration")# del param['num_iteration']
    
    early_stop = 5#param.get('early_stopping_rounds', 5)
    verbose_eval =100# param.get('verbose_eval', 100)
    
    # specify validations set to watch performance
    watchlist = [(dvalid, 'eval')]
    
    model = xgb.train(params=param, dtrain=dtrain,
                      num_boost_round=n_round, 
                       evals=watchlist, 
                      early_stopping_rounds=early_stop,
                      xgb_model=loaded_model,
                      verbose_eval=verbose_eval)
    dat = {'model': model,
           'param': param,
           'cols': X_regLmnt.columns,
           'N': Y_regLmnt.N[0]
           }
    
    return dat

def fit_per_element_bootstrap_gbm(X_regLmnt, Y_regLmnt, drivers, gbm_hyperparams, 
                                 n_bootstrap, path_pretrained_model = None,
                                 transferlearning = False, save_model = False):
    
    if transferlearning:
        # Load the pre-trained model on intergenic region
        with open(path_pretrained_model, 'rb') as file:
            loaded_model = pickle.load(file)
    
    
    elems = ["gc19_pc.ss", "enhancers",
             "gc19_pc.cds", "gc19_pc.promCore",
             "gc19_pc.5utr", "gc19_pc.3utr"] #"lncrna.ncrna", "lncrna.promCore",
    
    count_non_nans = pd.DataFrame()
    all_elems_ensemble_pred = pd.DataFrame()
    
    for elem in elems:
        path_save = gbm_hyperparams['path_save']
        
        print(f'**************** {elem} ******************')
        print('*******************************************')
        
        corr_values = []
        mse_values = []
        pred_samples = pd.DataFrame()
        for n in range(n_bootstrap):
            iteration = n+1
            print(f'bootstrap sample number: {iteration}')
            X_elem = split_by_element(X_regLmnt, elem)
            Y_elem = Y_regLmnt.loc[X_elem.index]
            
            X_samples_train, Y_samples_train, X_samples_test, Y_samples_test = generate_test_train_bootstrapSamples(X_elem, Y_elem, drivers)
            
            # train model and make prediction on a sample:
            if transferlearning:
                model_data = run_gbm_transferLearning(loaded_model, X_samples_train, 
                                                      Y_samples_train, gbm_hyperparams)
                
                # model_data = {'model': loaded_model,
                #         'param': gbm_hyperparams,
                #         'cols': X_regLmnt.columns,
                #         'N': Y_elem.N[0]
                #         }
                
            else:
                model_data = run_gbm(X_samples_train, Y_samples_train, gbm_hyperparams)
            
            
            samples_length = Y_samples_test.length
            pred_sample = predict_gbm(model_data, X_samples_test, samples_length)
            
            obs_sample = pd.DataFrame(Y_samples_test.nMut/(Y_samples_test.N * Y_samples_test.length))
            corr, p_value = spearmanr(pred_sample, obs_sample)
            print(f'{elem} obs-pred spearman corr: {corr}') 
            mse = mean_squared_error(obs_sample, pred_sample)
            print(f'{elem} obs-pred MSE : {mse}') 
            corr, p_value = spearmanr(pred_sample, obs_sample, nan_policy='omit')
            print(f'{elem} obs-pred spearman corr with omit: {corr}')
            
            
            # corr, p_value = spearmanr(gbm_pred, obs_sample)
            # print(f'GBM obs-pred spearman corr: {corr}') 
            # mse = mean_squared_error(obs_sample, gbm_pred)
            # print(f'GBM obs-pred MSE : {mse}') 
            print('----------------- 1 --------------------------')
            if save_model:
                
                os.makedirs(path_save, exist_ok=True)
                
                M = model_data['model']
                # Save the model using pickle
                save_path_model = f'{path_save}/model_{iteration}_{elem}.pkl'
                
                corr_values.append(corr)
                mse_values.append(mse)
                # Create a DataFrame to hold correlation and mse values
                result_df = pd.DataFrame({'Element': elem,
                                          'Correlation': corr_values, 'MSE': mse_values})
                # Save the DataFrame to a TSV file
                result_df.to_csv(f'{path_save}/correlation_mse_results_{elem}.tsv', sep='\t', index=False)
            
                print('----------------- 2 --------------------------')
                with open(save_path_model, 'wb') as f: 
                    pickle.dump(M, f)
                    
                
            print('-------------------------------------------')
            pred_samples = pd.concat([pred_samples, pred_sample], axis = 1)
            
            # count_non_nan: save the average number of runs for each pred
            count_non_nan = pd.DataFrame(pred_samples.iloc[:, :].count(axis=1))
            
            ensemble_preds = pd.DataFrame(pred_samples.mean(axis=1))
            tmp_ensemble_obs = Y_regLmnt.loc[ensemble_preds.index]
            ensemble_obs = tmp_ensemble_obs.nMut/(tmp_ensemble_obs.N * tmp_ensemble_obs.length)
            
            
            corr_ensemble, p_value = spearmanr(ensemble_preds, ensemble_obs)
            print(f'{elem} ensemble obs-pred spearman corr: {corr_ensemble} in {ensemble_preds.shape}  where total is {Y_elem.shape}') 
            mse = mean_squared_error(ensemble_obs, ensemble_preds)
            print(f'{elem} ensemble obs-pred MSE: {mse} ') 
            corr_ensemble, p_value = spearmanr(ensemble_preds, ensemble_obs, nan_policy='omit')
            print(f'{elem} ensemble obs-pred spearman corr with omit: {corr_ensemble} in {ensemble_preds.shape}  where total is {Y_elem.shape}') 
            
            print('--------------------------------')
            
        all_elems_ensemble_pred = pd.concat([all_elems_ensemble_pred, ensemble_preds], axis=0)
        count_non_nans = pd.concat([count_non_nans, count_non_nan], axis=0)
        
    print('############ Job Done ##############')
    return all_elems_ensemble_pred, count_non_nans



def fit_per_element_bootstrap_gbm2(elems, X_regLmnt, Y_regLmnt, drivers, NN_hyperparams, 
                                 n_bootstrap, gbm_hyperparams, path_pretrained_model = None,
                                 transferlearning = False):
    
    if transferlearning:
        # Load the pre-trained model on intergenic region
        with open(path_pretrained_model, 'rb') as file:
            loaded_model = pickle.load(file)
        
    
    all_GBM_preds = pd.read_csv('../external/output/GBM/GBM_predTest.tsv', 
                                sep='\t', index_col='binID')
    
    count_non_nans = pd.DataFrame()
    all_elems_ensemble_pred = pd.DataFrame()
    
    for elem in elems:
        
        print(f'**************** {elem} ******************')
        print('*******************************************')
        pred_samples = pd.DataFrame()
        for n in range(n_bootstrap):
            print(f'bootstrap sample number: {n+1}')
            X_elem = split_by_element(X_regLmnt, elem)
            Y_elem = Y_regLmnt.loc[X_elem.index]
            
            X_samples_train, Y_samples_train, X_samples_test, Y_samples_test = generate_test_train_bootstrapSamples(X_elem, Y_elem, drivers)
            length_elems = Y_samples_test.length
            
            # train model and make prediction on a sample:
            if transferlearning:
                model_data = run_gbm_transferLearning(loaded_model, X_samples_train, 
                                                      Y_samples_train, gbm_hyperparams)
                
                # model_data = {'model': loaded_model,
                #         'param': gbm_hyperparams,
                #         'cols': intergenicFeatures,
                #         'N': Y_elem.N[0]
                #         }
                
            else:
                model_data = run_gbm(X_samples_train, Y_samples_train, gbm_hyperparams)
            
            
            
            pred_sample = predict_gbm(model_data, X_samples_test, length_elems)
            
            gbm_pred = all_GBM_preds.loc[pred_sample.index]
            
            
            obs_sample = pd.DataFrame(Y_samples_test.nMut/(Y_samples_test.N * Y_samples_test.length))
            corr, p_value = spearmanr(pred_sample, obs_sample)
            print(f'{elem} obs-pred spearman corr: {corr}') 
            mse = mean_squared_error(obs_sample, pred_sample)
            print(f'{elem} obs-pred MSE : {mse}') 
            corr, p_value = spearmanr(pred_sample, obs_sample, nan_policy='omit')
            print(f'{elem} obs-pred spearman corr with omit: {corr}')
            
            corr, p_value = spearmanr(gbm_pred, obs_sample)
            print(f'GBM obs-pred spearman corr: {corr}') 
            mse = mean_squared_error(obs_sample, gbm_pred)
            print(f'GBM obs-pred MSE : {mse}') 
            
           
            
            pred_samples = pd.concat([pred_samples, pred_sample], axis = 1)
            
            # count_non_nan: save the average number of runs for each pred
            count_non_nan = pd.DataFrame(pred_samples.iloc[:, :].count(axis=1))
            
            ensemble_preds = pd.DataFrame(pred_samples.mean(axis=1))
            tmp_ensemble_obs = Y_regLmnt.loc[ensemble_preds.index]
            ensemble_obs = tmp_ensemble_obs.nMut/(tmp_ensemble_obs.N * tmp_ensemble_obs.length)
            pred_gbm_ensemble_ids = all_GBM_preds.loc[ensemble_preds.index]
            
            
            corr_ensemble, p_value = spearmanr(ensemble_preds, ensemble_obs)
            print(f'{elem} ensemble obs-pred spearman corr: {corr_ensemble} in {ensemble_preds.shape}  where total is {Y_elem.shape}') 
            mse = mean_squared_error(ensemble_obs, ensemble_preds)
            print(f'{elem} ensemble obs-pred MSE: {mse} ') 
            corr_ensemble, p_value = spearmanr(ensemble_preds, ensemble_obs, nan_policy='omit')
            print(f'{elem} ensemble obs-pred spearman corr with omit: {corr_ensemble} in {ensemble_preds.shape}  where total is {Y_elem.shape}') 
            
            corr_ensemble_gbm, p_value = spearmanr(pred_gbm_ensemble_ids, ensemble_obs)
            print(f'GBM ensemble obs-pred spearman corr: {corr_ensemble_gbm} ') 
            mse = mean_squared_error(ensemble_obs, pred_gbm_ensemble_ids)
            print(f'GBM ensemble obs-pred MSE: {mse} ') 
            
            
            print('--------------------------------')
            
        all_elems_ensemble_pred = pd.concat([all_elems_ensemble_pred, ensemble_preds], axis=0)
        count_non_nans = pd.concat([count_non_nans, count_non_nan], axis=0)
        
    print('############ Job Done ##############')
    return all_elems_ensemble_pred, count_non_nans



def eMET(sim_setting, path_ann_pcawg_IDs, path_pretrained_model, n_bootstrap = 100):
    
    X_regLmnt, Y_regLmnt = load_regulatory_elems(sim_setting)
    
    # restrict features to DP features
    new_ftrs = ['APOBEC3A', 'E001-DNAMethylSBS', 'E002-DNAMethylSBS',
                'E003-DNAMethylSBS', 
     'E004-DNAMethylSBS', 'E005-DNAMethylSBS', 'E006-DNAMethylSBS', 
     'E007-DNAMethylSBS', 'E008-DNAMethylSBS', 'E009-DNAMethylSBS', 
     'E010-DNAMethylSBS', 'E011-DNAMethylSBS', 'E012-DNAMethylSBS', 
     'E013-DNAMethylSBS', 'E014-DNAMethylSBS', 'E015-DNAMethylSBS',
     'E016-DNAMethylSBS', 'E017-DNAMethylSBS', 'E018-DNAMethylSBS', 
     'E019-DNAMethylSBS', 'E020-DNAMethylSBS', 'E021-DNAMethylSBS',
     'E022-DNAMethylSBS', 'E023-DNAMethylSBS', 'E024-DNAMethylSBS', 
     'E025-DNAMethylSBS', 'E026-DNAMethylSBS', 'E027-DNAMethylSBS',
     'E028-DNAMethylSBS', 'E029-DNAMethylSBS', 'E030-DNAMethylSBS', 
     'E031-DNAMethylSBS', 'E032-DNAMethylSBS', 'E033-DNAMethylSBS',
     'E034-DNAMethylSBS', 'E035-DNAMethylSBS', 'E036-DNAMethylSBS', 
     'E037-DNAMethylSBS', 'E038-DNAMethylSBS', 'E039-DNAMethylSBS', 
     'E040-DNAMethylSBS', 'E041-DNAMethylSBS', 'E042-DNAMethylSBS', 
     'E043-DNAMethylSBS', 'E044-DNAMethylSBS', 'E045-DNAMethylSBS', 
     'E046-DNAMethylSBS', 'E047-DNAMethylSBS', 'E048-DNAMethylSBS', 
     'E049-DNAMethylSBS', 'E050-DNAMethylSBS', 'E051-DNAMethylSBS', 
     'E052-DNAMethylSBS', 'E053-DNAMethylSBS', 'E054-DNAMethylSBS',
     'E055-DNAMethylSBS', 'E056-DNAMethylSBS', 'E057-DNAMethylSBS',
     'E058-DNAMethylSBS', 'E059-DNAMethylSBS', 'E061-DNAMethylSBS',
     'E062-DNAMethylSBS', 'E063-DNAMethylSBS', 'E065-DNAMethylSBS',
     'E066-DNAMethylSBS', 'E067-DNAMethylSBS', 'E068-DNAMethylSBS',
     'E069-DNAMethylSBS', 'E070-DNAMethylSBS', 'E071-DNAMethylSBS', 
     'E072-DNAMethylSBS', 'E073-DNAMethylSBS', 'E074-DNAMethylSBS', 
     'E075-DNAMethylSBS', 'E076-DNAMethylSBS', 'E077-DNAMethylSBS', 
     'E078-DNAMethylSBS', 'E079-DNAMethylSBS', 'E080-DNAMethylSBS', 
     'E081-DNAMethylSBS', 'E082-DNAMethylSBS', 'E083-DNAMethylSBS', 
     'E084-DNAMethylSBS', 'E085-DNAMethylSBS', 'E086-DNAMethylSBS', 
     'E087-DNAMethylSBS', 'E088-DNAMethylSBS', 'E089-DNAMethylSBS', 
     'E090-DNAMethylSBS', 'E091-DNAMethylSBS', 'E092-DNAMethylSBS', 
     'E093-DNAMethylSBS', 'E094-DNAMethylSBS', 'E095-DNAMethylSBS', 
     'E096-DNAMethylSBS', 'E097-DNAMethylSBS', 'E098-DNAMethylSBS', 
     'E099-DNAMethylSBS', 'E100-DNAMethylSBS', 'E101-DNAMethylSBS', 
     'E102-DNAMethylSBS', 'E103-DNAMethylSBS', 'E104-DNAMethylSBS', 
     'E105-DNAMethylSBS', 'E106-DNAMethylSBS', 'E107-DNAMethylSBS', 
     'E108-DNAMethylSBS', 'E109-DNAMethylSBS', 'E110-DNAMethylSBS', 
     'E111-DNAMethylSBS', 'E112-DNAMethylSBS', 'E113-DNAMethylSBS', 
     'E114-DNAMethylSBS', 'E115-DNAMethylSBS', 'E116-DNAMethylSBS', 
     'E117-DNAMethylSBS', 'E118-DNAMethylSBS', 'E119-DNAMethylSBS', 
     'E120-DNAMethylSBS', 'E121-DNAMethylSBS', 'E122-DNAMethylSBS', 
     'E123-DNAMethylSBS', 'E124-DNAMethylSBS', 'E125-DNAMethylSBS', 
     'E126-DNAMethylSBS', 'E127-DNAMethylSBS', 'E128-DNAMethylSBS', 
     'E129-DNAMethylSBS'
     # , 'primates_phastCons46way', 
     # 'primates_phyloP46way', 'vertebrate_phastCons46way'
     ]
    #new_ftrs = []
    columns_to_exclude = [col for col in new_ftrs if col in X_regLmnt.columns]
    X_regLmnt = X_regLmnt.drop(columns=columns_to_exclude, errors='ignore') 

    # reorder X_regLmnt based on intergenic features to be compatible for model loading and transfer learning
    # np.save('tmp_intergenicFeaturesOrder.npy', intergenicFeatures)
    # intergenicFeatures = np.load('tmp_intergenicFeaturesOrder.npy', allow_pickle=True)
    # X_regLmnt = X_regLmnt[intergenicFeatures]

    drivers = get_identified_drivers(path_ann_pcawg_IDs, based_on = 'all')
    print(X_regLmnt.shape)

    base_dir = sim_setting['base_dir']
    models = sim_setting['models']
    model_name = list(models.keys())[0]
    m = models[model_name]
    name = m['save_name']
    gbm_hyperparams = m['Args']
    save_path_model = f'{base_dir}/{model_name}/'
    gbm_hyperparams['path_save'] = f'{save_path_model}models_interval/'
    Nr_pair_acc = sim_setting['Nr_pair_acc']






    # pred_ensemble_bootstraps, n_runs_per_pred = fit_per_element_bootstrap_gbm(X_regLmnt, Y_regLmnt, drivers, gbm_hyperparams, 
    #                                   n_bootstrap)
    pred_ensemble_bootstraps, n_runs_per_pred = fit_per_element_bootstrap_gbm(X_regLmnt, Y_regLmnt, drivers, gbm_hyperparams, 
                                      n_bootstrap, path_pretrained_model = path_pretrained_model,
                                      transferlearning = True)


    obs_rates = Y_regLmnt.nMut/(Y_regLmnt.length*Y_regLmnt.N)
    obs_pred_rates = pd.concat([obs_rates, pred_ensemble_bootstraps], axis=1)
    obs_pred_rates = pd.concat([obs_pred_rates, n_runs_per_pred], axis=1)
    obs_pred_rates.columns = ['obs_rates', 'predRate', 'n_runs_per_pred']
    obs_pred_rates['n_runs_per_pred'] = obs_pred_rates['n_runs_per_pred'].fillna(0)


    os.makedirs(f'{base_dir}/{model_name}/', exist_ok=True)
    obs_pred_rates.to_csv(f'{base_dir}/{model_name}/{model_name}_{n_bootstrap}_predTest.tsv', sep = '\t')

    assessment = assess_model(obs_pred_rates.predRate, obs_pred_rates.obs_rates, 
                  Nr_pair_acc, model_name, per_element=True)

    assessment.to_csv(f'{base_dir}/{model_name}/{model_name}_ensemble_bootstraps{n_bootstrap}_assessment.tsv', sep = '\t')

import shutil
import configparser
from simulation_settings import config_get

def config_save_eMET(sim_file, change_dir_save = ''):
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
        


def one_group_importance_eMET(sim_file, path_ann_pcawg_IDs, path_pretrained_model, n_bootstrap = 100): #path_pretrained_model = f'../external/BMR/output/featureImportance/GBM_{feature_category_name}/GBM_{feature_category_name}_model.pkl'
    

    # path_pretrained_model = '../external/BMR/output/featureImportance/GBM_feature_group/GBM_feature_group_model.pkl'

    sim_setting = load_sim_settings(sim_file)

    X_regLmnt, Y_regLmnt = load_regulatory_elems(sim_setting)




    drivers = get_identified_drivers(path_ann_pcawg_IDs, based_on = 'all')
    sim_base_dir = sim_setting['base_dir']

    models = sim_setting['models']
    model_name = list(models.keys())[0]
    m = models[model_name]
    # name = m['save_name']
    gbm_hyperparams = m['Args']
    Nr_pair_acc = sim_setting['Nr_pair_acc']
    
    categories = [ 'nucleotide content', 'DNA_accessibility', 'HiC', 'Epigenetic_mark', 'RNA_expression', 'Replication_timing', 'conservation']

    for feature_category in categories:
        
        print(feature_category)
        
        
        
        
        config_save_eMET(sim_file, feature_category)
        
        ftrs = get_features_category(category= [feature_category])
        
        X_regLmnt_ftrs = X_regLmnt.loc[:, ftrs]
        print(X_regLmnt_ftrs.shape)
        
        base_dir = f'{sim_base_dir}{feature_category}'
        
        save_path_model = f'{base_dir}/{model_name}/'
        gbm_hyperparams['path_save'] = f'{save_path_model}models_interval/'
        
        if feature_category == 'nucleotide content':
            feature_category_name = 'nucleotideContext'
        else:
           feature_category_name = feature_category
           
        path_intergenic_model = path_pretrained_model.replace('feature_group', feature_category_name)
        
        pred_ensemble_bootstraps, n_runs_per_pred = fit_per_element_bootstrap_gbm(X_regLmnt_ftrs, Y_regLmnt, drivers, gbm_hyperparams, 
                                          n_bootstrap, path_intergenic_model,
                                          transferlearning = True, save_model=True)
        obs_rates = Y_regLmnt.nMut/(Y_regLmnt.length*Y_regLmnt.N)
        obs_pred_rates = pd.concat([obs_rates, pred_ensemble_bootstraps], axis=1)
        obs_pred_rates = pd.concat([obs_pred_rates, n_runs_per_pred], axis=1)
        obs_pred_rates.columns = ['obs_rates', 'predRate', 'n_runs_per_pred']
        obs_pred_rates['n_runs_per_pred'] = obs_pred_rates['n_runs_per_pred'].fillna(0)
        
        os.makedirs(f'{base_dir}/{model_name}/', exist_ok=True)
        obs_pred_rates.to_csv(f'{base_dir}/{model_name}/{model_name}_{n_bootstrap}_predTest.tsv', sep = '\t')
        
        assessment = assess_model(obs_pred_rates.predRate, obs_pred_rates.obs_rates, 
                      Nr_pair_acc, model_name, per_element=True)
        
        assessment.to_csv(f'{base_dir}/{model_name}/{model_name}_ensemble_bootstraps{n_bootstrap}_assessment.tsv', sep = '\t')


def save_importance_ratio_dfs(sim_setting, path_full_model_eMET, n_bootstrap):
    
    categories = [ 'nucleotide content', 'HiC', 
                  'DNA_accessibility', 'Epigenetic_mark', 'RNA_expression', 'Replication_timing',
                  'conservation'] 
    sim_base_dir = sim_setting['base_dir']
    models = sim_setting['models']
    model_name = list(models.keys())[0]
    
    full_model_df = pd.read_csv(path_full_model_eMET, 
                                sep=",", index_col=0, skipinitialspace=True)
        
    for category in categories:
                   
        base_dir = f'{sim_base_dir}{category}'
        
            
        path_one_group_ass = f'{base_dir}/{model_name}/{model_name}_ensemble_bootstraps{n_bootstrap}_assessment.tsv'
         
        one_group_df = pd.read_csv(path_one_group_ass, sep="\t", index_col=0)
        
        corr_model = f'corr_{model_name}'
        
        # Extract correlation values from the dataframes
        one_group_corr = one_group_df.loc[corr_model]
        full_model_corr = full_model_df.loc[corr_model]
        
        # Create a dataframe to store the ratios
        ratios_df = pd.DataFrame(columns=["Element", "Ratio"])
        
        # Calculate and store the ratios for each element type
        for element in one_group_corr.index:
            if element in full_model_corr.index:
                ratio = one_group_corr[element] / full_model_corr[element]
                ratios_df.loc[len(ratios_df)] = [element, ratio]
               
        # Save the ratios to a new TSV file
        ratios_df.to_csv(f'{base_dir}/{model_name}/{model_name}importanceRatios_{category}.tsv', sep="\t", index=False)
        
    ###############################################################################
    all_ratio_dfs = pd.DataFrame(columns=["Element", "Ratio", "feature_category"])
    
    for category in categories:
        
        ratio_df = pd.read_csv(f'{sim_base_dir}{category}/{model_name}/{model_name}importanceRatios_{category}.tsv', sep="\t")
        ratio_df["feature_category"] = category
        
        all_ratio_dfs = pd.concat([all_ratio_dfs, ratio_df], axis = 0)
        
    all_ratio_dfs.to_csv(f"{sim_base_dir}/importanceRatios.csv", sep=",")   

