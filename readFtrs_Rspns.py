""" Data input and output module for DNN.

Input file types: X (tsv), y (tsv)

"""
import os
import logging
import pandas as pd
import numpy as np
import warnings
import h5py
import ast
from collections import namedtuple
# from sklearn.preprocessing import RobustScaler
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger('IO')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def read_feature(path, included_bins, use_features=None):
    """Read X (features) table     
    """
    if path.lower().endswith(('.h5', '.hdf5')):
        # open the hdf5
        f = h5py.File(path, 'r')
        X = f['/X/block0_values'] # feature values
        
        ftr_names = f['/X/block0_items'][:] # feature names
        ftr_names = np.array([val.decode('utf-8') for val in ftr_names])
        
        bins = f['/X/axis1'][:]
        binID = np.array([val.decode('utf-8') for val in bins])

        # filter bins to include only those in included_bins
        # included_bins = set(included_bins)
        # bin_mask = np.array([b in included_bins for b in bins])

        # select subset of data based on the bin_mask
        X = pd.DataFrame(X, index=binID, columns=ftr_names)
        X = X.loc[included_bins]
        # just include a subset of features
        if use_features is not None:
            X = X[use_features]
            
        f.close()
        
    elif path.lower().endswith(('.tsv')):
        X = pd.read_csv(path, sep='\t', header=0, index_col='binID')
        if use_features is not None:
            X = X[use_features]
            
        X = X.loc[included_bins]
    
    return X

def read_response(path):
    """Read y (response) table in TSV format.
    
    y should have exactly five columns: binID, length, nMut, nSample, N
    
    Args:
        path (str): Path to the file.
    
    Returns:
        pd.df: A panda DF indexed by binID.
        
    """
    y = pd.read_csv(path, sep='\t', header=0, index_col='binID',
                    usecols=['binID', 'length', 'nMut', 'nSample', 'N'])
    # y['obsRate'] = np.log((y['nMut']) / (y['length']))
    y['obsRates'] = y.nMut/(y.length * y.N)
    y['offset'] = y['N'] * y['length']
    # sanity check
    assert len(y.index.values) == len(y.index.unique()), "binID in response table is not unique."
    return y

def split_by_element(df, element_type):
    df_element = df.loc[df.index.str.contains(element_type)]
    return df_element




def split_test_train(X_cmplt, Y_cmplt, test_size=0.2):
    S = int(X_cmplt.shape[0]*test_size)
    bins = np.random.choice(X_cmplt.index.unique(), size=S, replace=False)
    X_validat = X_cmplt.loc[bins]
    Y_validate = Y_cmplt.loc[bins]
    
    Y_train = Y_cmplt.drop(index=bins)
    X_train = X_cmplt.drop(index=bins)
    
    DataTuple = namedtuple('DataTuple', ['X_train', 'Y_train', 'X_validat', 'Y_validate'])
    
    # create instance of named tuple
    data = DataTuple(X_train=X_train, Y_train=Y_train,
                     X_validat = X_validat, Y_validate = Y_validate)
    return data


def scale_train(X_cmplt):
    
    print("train data is scaling please wait")
    train_sd = X_cmplt.std()
    train_mean = X_cmplt.mean()
    
    X_standardized = (X_cmplt - train_mean) / train_sd
    return X_standardized, train_mean, train_sd

def scale_test(X, train_mean, train_sd):
    
    X_standardized = (X - train_mean) / train_sd
    
    return X_standardized


def load_data(path_X, path_Y, use_features = None):
    # Generate the feature and response table for all of the whiteListed bins and the feature superset
    print("====== Please wait! Data is loading... ======")
    Y_cmplt = read_response(path_Y)
    included_bins = Y_cmplt.index[:]
    X_cmplt = read_feature(path_X, included_bins, use_features)
    print("====== Data was loaded successfuly =======")
    
    return X_cmplt, Y_cmplt



def create_TestTrain_SingleSource(path_X, path_Y, test_size, scale = True,
                                  use_features = None):
    X_cmplt, Y_cmplt = load_data(path_X, path_Y)
    
    if (X_cmplt.index == Y_cmplt.index).all():
        print("The index values of X match the index values of Y.")
    else:
        raise ValueError("The index values of X do not match the index values of Y.")
    
    if use_features is not None:
        X_cmplt = X_cmplt[use_features]
    
    if scale:
        X_cmplt, meanSc, sdSc = scale_train(X_cmplt)
    
    X_train, Y_train, X_val, Y_val = split_test_train(X_cmplt, Y_cmplt, test_size)
    
    return X_train, Y_train, X_val, Y_val


def create_TestTrain_TwoSources(path_X_train, path_Y_train, path_X_test,
                                path_Y_test, scale = True, use_features = None):
    
    X_train, Y_train = load_data(path_X_train, path_Y_train, use_features)
    
    X_test, Y_test = load_data(path_X_test, path_Y_test, use_features)
    
    # reorder test columns based on train columns
    X_test = X_test[X_train.columns]
    
    if use_features is not None:
        X_train = X_train[use_features]
        X_test = X_test[use_features]
    
    if scale:
        X_train, meanSc, sdSc = scale_train(X_train)
        X_test = scale_test(X_test, meanSc, sdSc)
    
    return X_train, Y_train, X_test, Y_test



def save_preds_tsv(fitted_Model, path_save, save_name, iteration = None):
    
    
    os.makedirs(f'{path_save}/{save_name}/rep_train_test/',
                exist_ok= True)
    
    # calculate predictions:
    predRates_train = fitted_Model.predRates_train
    
    # save the predicted values DataFrame to a file
    path_file_tr = f'{path_save}/{save_name}/{save_name}_predTrain.tsv'
    path_file_test = f'{path_save}/{save_name}/{save_name}_predTest.tsv'
    
    if iteration is not None:
       path_file_test = f'{path_save}/{save_name}/rep_train_test/{save_name}_predTest{iteration}.tsv'
        
    if iteration is None: 
        predRates_train.to_csv(path_file_tr,
                           sep = '\t')
    
    
    predRates_test = fitted_Model.predRates_test
    predRates_test.to_csv(path_file_test,
                          sep = '\t')




############
import subprocess

def get_latest_commit_hash():
    try:
        # Run the Git command and capture the output
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('ascii')
        return git_hash
    except Exception:
        return None


def paste0(string, Range):
    texts = [string + str(num) for num in Range]

    return np.array(texts)


def load_regulatory_elems(sim_setting):
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
        
        Y_test = Y_test[Y_test['nMut'] != 0]
        X_test = X_test.loc[Y_test.index]
    
    if (Y_test.index != X_test.index).all():
        raise ValueError('X_test and Y_test indexes are not the same')
    
    return X_test, Y_test


def create_100k_sets(data_path, output_folder):
    os.makedirs(output_folder, exist_ok= True)
    all_Y_intergenic = read_response(data_path)
    indices = all_Y_intergenic.index.unique()
    
    idx_indices = np.random.choice(list(indices), size=100000, replace=False)
    intergenic100k = all_Y_intergenic.loc[idx_indices]
    mask= all_Y_intergenic[~all_Y_intergenic.index.isin(idx_indices)]
    validation100k_idx = np.random.choice(list(mask.index.unique()), size=100000, replace=False)
    validation100k = mask.loc[validation100k_idx]
    
    intergenic100k.to_csv(f'{output_folder}/Pan_Cancer_100k_intergenic.tsv', sep='\t')
    validation100k.to_csv(f'{output_folder}/Pan_Cancer_100kval_intergenic.tsv', sep='\t')


###########################################3
import tensorflow as tf
def set_gpu_memory_limit(gpu_fraction):
    # Check if TensorFlow is built with GPU support
    if tf.test.is_built_with_cuda():
        # Create a TensorFlow session with GPU memory growth enabled
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)
        

def read_fi(path, cutoff=0.8):
    """Read feature importance table in TSV format.

    Feature importance table must contain two columns: name and importance

    Args:
        path (str): path to the file.
        cutoff (float): cutoff of feature selection.

    Returns:
        list: useful features. Return None if path is None.

    """
    fi = pd.read_csv(path, sep='\t', header=0, index_col='name',
                     usecols=['name', 'importance'])
    assert len(fi.index.values) == len(fi.index.unique()), \
        "Feature name in feature importance table is not unique."
    keep = (fi.importance >= cutoff).values
    use_features = fi.loc[keep]
    use_features = use_features.index.values
    
    return use_features
