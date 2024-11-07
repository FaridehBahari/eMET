import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from readFtrs_Rspns import save_preds_tsv
import configparser
def build_GBM_params(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

    # Load the hyperparameters from the config file
    params = {
        'max_depth': config.getint('train', 'max_depth'),
        'eta': config.getfloat('train', 'eta'),
        'subsample': config.getfloat('train', 'subsample'),
        'nthread': config.getint('train', 'nthread'),
        'objective': config.get('train', 'objective'),
        'max_delta_step': config.getfloat('train', 'max_delta_step'),
        'eval_metric': config.get('train', 'eval_metric'),
        'num_iteration': config.getint('train', 'num_iteration'),
        'tree_method': config.get('train', 'tree_method'), # 'gpu_hist' ,#if use_gpu else 'hist',
        'gpu_id': 0#gpu_id if use_gpu else -1  # -1 means use CPU

    }
    return params

def gbm_model_info(save_name, *args):
    params = build_GBM_params(args[0])
    model_dict = {"save_name" : save_name,
                  "Args" : params,
                  "run_func": run_gbm,
                  "predict_func": predict_gbm,
                  "save_func": save_gbm,
                  "check_file_func": check_file_gbm
                  }
    
    return model_dict


def run_gbm(X_train, Y_train, param):
    
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train,
                                                        test_size=0.12, 
                                                        shuffle=True)
    if ((X_train.index != Y_train.index).all()) or ((X_valid.index != Y_valid.index).all()):
            ValueError("The index values of X doesnt match the index values of Y.")
    
    # calculate base margin
    offset_train = np.array(np.log(Y_train.length+1/Y_train.N) + np.log(Y_train.N))
    offset_valid = np.array(np.log(Y_valid.length+1/Y_valid.N) + np.log(Y_valid.N))
    
    ftr_names = X_train.columns.values
    
    if param['tree_method'] == 'gpu_hist':
        ftr_names = X_train.columns.values.tolist()
        
    dtrain = xgb.DMatrix(data=X_train, label=Y_train.nMut.values, feature_names=ftr_names)
    dvalid = xgb.DMatrix(data=X_valid, label=Y_valid.nMut.values, feature_names=ftr_names)
    
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
    
    model = xgb.train(params=param, dtrain=dtrain, num_boost_round=n_round, 
                      evals=watchlist, 
                      early_stopping_rounds=early_stop, 
                      verbose_eval=verbose_eval)
    dat = {'model': model,
           'param': param,
           'cols': X_train.columns,
           'N': Y_train.N[0]
           }
    
    return dat

def nMut_2_rate(nMut, N, length):
    rates = nMut/(length * N)
    return rates





def predict_gbm(model, X_test, length_elems):
    cols = model['cols']
    X_test = X_test[cols]
    M = model['model']
    param = model['param']
    N = model['N']
    
    # if model was saved
    # model = xgb.Booster()
    # model.load_model(path_model)
    
    # prepare test data for prediction
    dtest = xgb.DMatrix(X_test)
    dtest.set_base_margin(np.array(np.log(length_elems+1/N) + np.log(N)))
    M.set_param(param)  # Bypass a bug of dumping without max_delta_step
    
    prediction = M.predict(dtest)
    prediction =  nMut_2_rate(prediction, model['N'], length_elems)
    prediction_df = pd.DataFrame({'predRate': prediction.ravel()}, 
                                 index=X_test.index)
    return prediction_df

def save_gbm(fitted_Model, path_save, save_name, iteration = None, save_model = True): 
    
    save_preds_tsv(fitted_Model, path_save, save_name, iteration)
    
    if save_model:
        M = fitted_Model.model['model']
        # Save the model using pickle
        if iteration is not None:
            save_path_model = f'{path_save}/{save_name}/rep_train_test/{save_name}_model_{iteration}.pkl'
        else:
            save_path_model = f'{path_save}/{save_name}/{save_name}_model.pkl'
        with open(save_path_model, 'wb') as f: 
            pickle.dump(M, f)
    

        
        
def check_file_gbm(path_save, save_name):
    file_name = f'{path_save}/{save_name}/{save_name}_model.pkl'
    return file_name  
 