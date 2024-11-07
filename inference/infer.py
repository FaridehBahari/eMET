# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 18:23:03 2023

@author: Farideh
dispersion_test, negbinom_test, burden_test, and bh_fdr functions are from DriverPower codes.
"""
import numpy as np
import os
import logging
from scipy.stats import binom_test, nbinom #DeprecationWarning: 'binom_test' is deprecated in favour of 'binomtest' from version 1.7.0 and will be removed in Scipy 1.12.0.
from sklearn.utils import resample
import statsmodels as sm
import sys
import pandas as pd 
from statsmodels.regression.linear_model import OLS
from statsmodels.sandbox.stats.multicomp import multipletests
logger = logging.getLogger('INFER')
from simulation_settings import load_sim_settings_perBatchPerf
from readFtrs_Rspns import read_response
from performance.assessModels import read_pred

def dispersion_test(yhat, y, k=100):
    """ Implement the regression based dispersion test with k re-sampling.
    Args:
        yhat (np.array): predicted mutation count
        y (np.array): observed mutation count
        k (int):
            
    Returns:
        float, float: p-value, theta
        
    """
    theta = 0
    pval = 0
    for i in range(k):
        y_sub, yhat_sub = resample(y, yhat, random_state=i)
        # (np.power((y - yhat), 2) - y) / yhat for Poisson regression
        aux = (np.power((y_sub - yhat_sub), 2) - yhat_sub) / yhat_sub
        mod = OLS(aux, yhat_sub)
        res = mod.fit()
        theta += res.params[0]
        pval += res.pvalues[0]
    theta = theta/k
    pval = pval/k
    return pval, theta





def negbinom_test(x, mu, theta, offset):
    """ Test with negative binomial distribution
    
    Convert mu and theta to scipy parameters n and p:
        
    p = 1 / (theta * mu + 1)
    n = mu * p / (1 - p)
    
    Args:
        x (float): observed number of mutations (or gmean).
        mu (float): predicted number of mutations (mean of negative binomial distribution).
        theta (float): dispersion parameter of negative binomial distribution.
    Returns:
        float: p-value from NB CDF. pval = 1 - F(n<x)
    """
    if offset == 1:  # element with 0 bp
        return 1
    p = 1 / (theta * mu + 1)
    n = mu * p / (1 - p)
    pval = 1 - nbinom.cdf(x, n, p, loc=1)
    return pval


def burden_test(count, pred, offset, test_method, pval_dispersion, theta, s):
    """ Perform burden test.
    
    Args:
        count:
        pred:
        offset:
        test_method:
        model:
        s:
        use_gmean:
    Returns:
    """
    # if pval_dispersion > 0.05:
    #     print('binomial test is used for calculating p-value' )
    # else:
    #     print('negative_binomial test is used for calculating p-value')
    if test_method == 'auto':
        test_method = 'binomial' if pval_dispersion > 0.05 else 'negative_binomial'
    if test_method == 'negative_binomial':
        logger.info('Using negative binomial test with s={}, theta={}'.format(s, theta))
        theta = s * theta
        pvals = np.array([negbinom_test(x, mu, theta, o)
                          for x, mu, o in zip(count, pred, offset)])
    elif test_method == 'binomial':
        logger.info('Using binomial test')
        pvals = np.array([binom_test(x, n, p, 'greater')
                          for x, n, p in zip(count, offset,
                                             pred/offset)])
    else:
        logger.error('Unknown test method: {}. Please use binomial, negative_binomial or auto'.format(test_method))
        sys.exit(1)
    return pvals


def bh_fdr(pvals):
    """ BH FDR correction
    Args:
        pvals (np.array): array of p-values
    Returns:
        np.array: array of q-values
    """
    return multipletests(pvals, method='fdr_bh')[1]

##############################################

def find_param_ini(directory_path, cancer_type = None):
    
    if cancer_type != None:
        if cancer_type == 'Pancan-no-skin-melanoma-lymph':
            setting_config = 'sim_setting_iDriver.ini'
        else:
            setting_config = f'sim_setting_iDriver_{cancer_type}.ini'
    else:
        setting_config = 'sim_setting.ini'
    
    second_ini = None
    for filename in os.listdir(directory_path):
        if filename.endswith('.ini') and filename != setting_config:
            second_ini = filename
            break
    return second_ini


def get_pred_path(directory_path, save_name):
    # Construct the base filename
    base_filename = f"{directory_path}/{save_name}_predTest.tsv"
    
    # If the base filename doesn't exist, return it directly
    if not os.path.exists(base_filename):
        count = 1
        while True:
            base_filename = f"{directory_path}/{save_name}_{count}_predTest.tsv"
            if os.path.exists(base_filename):
                return base_filename
            count += 1
    else:
        return base_filename


def perform_burdenTest(dir_path, cancer_type = None):
    
    if cancer_type != None:
        if cancer_type == 'Pancan-no-skin-melanoma-lymph':
            setting_config = 'sim_setting_iDriver.ini'
        else:
            setting_config = f'sim_setting_iDriver_{cancer_type}.ini'
        param_config = find_param_ini(dir_path, cancer_type)
    else:
        setting_config = 'sim_setting.ini'
        param_config = find_param_ini(dir_path)
    
    print(param_config)
    
    sim_setting = load_sim_settings_perBatchPerf(dir_path, setting_config,
                                                 param_config)
    save_name = list(sim_setting['models'].keys())[0]
    # sim_params = sim_setting['models'][save_name]
    # predict_func = sim_params['predict_func']
    base_dir = sim_setting['base_dir']
    directory_path = f'{base_dir}/{save_name}/'
    
    path_obs = sim_setting['path_Y_test']
    path_pred = get_pred_path(directory_path, save_name)
    print(path_pred)
    
    Y_obs = read_response(path_obs)
    Y_pred = read_pred(path_pred)
    
    # merge the data frames based on their indexes using merge()
    y = pd.merge(Y_obs, Y_pred, left_index=True,
                           right_index=True, how='inner')
    # Exclude rows where the index contains the 'lncrna' pattern
    y = y[~y.index.str.contains('lncrna', case=False, na=False)]
    #y['predRate'].isna().sum()
    y['nPred'] = (y.predRate*y.length*y.N)  
    use_gmean = True
    count = np.sqrt(y.nMut * y.nSample) if use_gmean else y.nMut
    offset = y.length * y.N + 1
    #test_method = 'negative_binomial'
    s = 1
    pval_dispersion, theta = dispersion_test(y.nPred, y.nMut, k=100)
    # y['raw_p_nBinom'] = burden_test(count, y.nPred, offset, 'negative_binomial',
    #                          pval_dispersion, theta, s)
    # y['raw_q_nBinom'] = bh_fdr(y.raw_p_nBinom)
    
    
    y['p_value'] = burden_test(count, y.nPred, offset, 'binomial',
                             pval_dispersion, theta, s)
    y['fdr'] = bh_fdr(y.p_value)
    
    
    inference_dir = f'{base_dir}/{save_name}/inference/'
    os.makedirs(inference_dir, exist_ok= True)
    y.to_csv(f'{inference_dir + save_name}_inference.tsv',
                      sep='\t')
    print('****************')