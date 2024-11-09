from models.runBMR_functions import  config_save
from simulation_settings import load_sim_settings
from models.eMET_functions import eMET
import time
import argparse

parser = argparse.ArgumentParser(description='Run eMET for rate prediction')
parser.add_argument('sim_file', type=str, help='the path to the simulation setting config')
parser.add_argument('path_pretrained_model', type=str, help='the path to the intergenic pretrained model')
parser.add_argument('n_bootstrap', type=int, help='the number of bootstrap samples')
args = parser.parse_args() 


sim_file = args.sim_file
path_pretrained_model = args.path_pretrained_model
n_bootstrap = args.n_bootstrap


st_time = time.time()
sim_setting = load_sim_settings(sim_file)
config_save(sim_file)




sim_setting = load_sim_settings(sim_file)
config_save(sim_file)



eMET(sim_setting, path_pretrained_model, n_bootstrap)
