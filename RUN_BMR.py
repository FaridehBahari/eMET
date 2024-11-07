import time
import os
import platform
import argparse
from readFtrs_Rspns import set_gpu_memory_limit
from models.runBMR_functions import  RUN_BMR, load_data_sim, config_save, repeated_train_test
from performance.assessModels import assess_models
from simulation_settings import load_sim_settings


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpu_fraction = 0.2
set_gpu_memory_limit(gpu_fraction)


if platform.system() == 'Windows':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    sim_file = 'configs/quick/sim_setting.ini'
    # sim_file = 'configs/rate_based/sim_setting.ini'
else:
    parser = argparse.ArgumentParser(description='Train different models for rate prediction on intergenic bins')
    parser.add_argument('sim_file', type=str, help='the path to the simulation setting config')
    args = parser.parse_args()
    sim_file = args.sim_file
    
st_time = time.time()
sim_setting = load_sim_settings(sim_file)
config_save(sim_file)

X_train, Y_train, X_test, Y_test = load_data_sim(sim_setting, category = [])

print(X_train.shape)
print(X_test.shape)
RUN_BMR(sim_setting, X_train, Y_train, X_test, Y_test, make_pred=True)

print('************')

assess_models(sim_setting)
end_t = time.time()

print('****** BMR model generated and performance on regulatory elements was assessed ******')
print(f'total time = {end_t - st_time} seconds')


