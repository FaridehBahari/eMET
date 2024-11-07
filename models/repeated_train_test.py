import pandas as pd
import os
import numpy as np
import re
from scipy.stats import ttest_rel

def get_metric_value(df, metric):
    metric_pattern = re.compile(f'{metric}_[A-Za-z0-9]+')
    
    # Convert index values to strings and exclude 'nan'
    index_values = [str(index) for index in df.index if not pd.isna(index)]
    
    metric_index = [index for index in index_values if metric_pattern.match(index)]

    if metric_index:
        metric_value = df.loc[metric_index[0], 'metric_value']
        return metric_value
    else:
        print(f"{metric} not found in the DataFrame.")
        return None


def extract_metric_values_model(dir_path, metric):
    # Define a regular expression pattern to match files with '_M\d_assessment.tsv' format
    file_pattern = re.compile(r'_M(\d+)_assessment\.tsv')

    # Get all files in the directory matching the pattern
    files = [f for f in os.listdir(dir_path) if file_pattern.search(f)]
    
    metric_values = []
    # Iterate through each file
    for file in files:
        file_path = os.path.join(dir_path, file)

        # Read the TSV file into a DataFrame
        df = pd.read_csv(file_path, sep='\t', index_col=0, header=None, 
                         names=['metric_value'])

        # Extract the metric value based on the provided metric argument
        metric_value = get_metric_value(df, metric)
        metric_values.append(metric_value)
        
    return metric_values



def Exp_Var_performance_onValSets(dir_path, metric):
    
    metric_values = extract_metric_values_model(dir_path, metric) 
    
    # Convert the list to a NumPy array
    metric_values_array = np.array(metric_values)

    # Convert the NumPy array to floating-point numbers
    metric_values_array = metric_values_array.astype(float)

    # Calculate the mean and variance
    mean_value = metric_values_array.mean()
    var_value =  metric_values_array.var()
    
    return mean_value, var_value

def save_metrics_summary(dir_path):
    result = []
    metrics = ['corr', 'acc', 'mse', 'made']
    
    for performance_metric in metrics:
        Exp, Var = Exp_Var_performance_onValSets(dir_path, performance_metric)
        
        # Append the results to the list
        result.append({
            'Metric': performance_metric,
            'Mean': Exp,
            'Variance': Var
        })
        
    # Create the DataFrame
    result_df = pd.DataFrame(result).set_index('Metric').pivot_table(index=None, columns='Metric')
    
    # Save the DataFrame to a CSV file in the parent directory
    parent_dir_path = os.path.dirname(os.path.abspath(dir_path))
    output_path = os.path.join(parent_dir_path, 'model_metrics_summary.tsv')
    result_df.to_csv(output_path, sep = '\t')
    print(f"Metrics summary saved to {output_path}")




def compare_two_models(dir_path1, dir_path2, metric):
    M1_metrics = extract_metric_values_model(dir_path1, metric)
    M2_metrics = extract_metric_values_model(dir_path2, metric)
    
    t_statistic, p_value = ttest_rel(M1_metrics, M2_metrics)
    
    return t_statistic, p_value
        
#############################################################################
# # EXAMPLE
# # calculate the average and variance of repeated models performance on validation sets:
# dir_path = '../external/BMR/output/quick/GBM/rep_train_test/'
# performance_metric = 'corr'  
# cor_meanVar = Exp_Var_performance_onValSets(dir_path, performance_metric)
# print(cor_meanVar)


# # save the model metrics summary:
# save_metrics_summary(dir_path)


# compare_two_models(dir_path1, dir_path2, performance_metric)
