from models.runBMR_functions import  config_save
from simulation_settings import load_sim_settings
from models.eMET_functions import eMET
import time
import argparse

parser = argparse.ArgumentParser(description='Run eMET for rate prediction')
parser.add_argument('sim_file', type=str, help='the path to the simulation setting config')
parser.add_argument('path_ann_pcawg_IDs', type=str, help='the path to the annotated bin IDs')
parser.add_argument('path_pretrained_model', type=str, help='the path to the intergenic pretrained model')
parser.add_argument('n_bootstrap', type=int, help='the number of bootstrap samples')
args = parser.parse_args() 


sim_file = args.sim_file
path_ann_pcawg_IDs = args.path_ann_pcawg_IDs
path_pretrained_model = args.path_pretrained_model
n_bootstrap = args.n_bootstrap


st_time = time.time()
sim_setting = load_sim_settings(sim_file)
config_save(sim_file)



# path_ann_pcawg_IDs = '../external/BMR/procInput/ann_PCAWG_ID_complement.csv'
# sim_file = 'configs/rate_based/sim_setting.ini'
# path_pretrained_model = '../external/BMR/output/with_RepliSeq_HiC/bin_size_effect/var_size/GBM/GBM_model.pkl'


sim_setting = load_sim_settings(sim_file)
config_save(sim_file)



eMET(sim_setting, path_ann_pcawg_IDs, path_pretrained_model, n_bootstrap)
