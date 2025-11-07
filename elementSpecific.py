from models.runBMR_functions import  config_save
from simulation_settings import load_sim_settings
from models.eMET_functions import elemSp
import time
import argparse

parser = argparse.ArgumentParser(description='Run CDS-specific for rate prediction')
parser.add_argument('sim_file', type=str, help='the path to the simulation setting config')
parser.add_argument('n_bootstrap', type=int, help='the number of bootstrap samples')
parser.add_argument('--elems', nargs='+', default=None,
                    help='List of elements (e.g., gc19_pc.cds gc19_pc.prom). '
                         'If not specified, the default inside elemSp() will be used.')
args = parser.parse_args() 


sim_file = args.sim_file
n_bootstrap = args.n_bootstrap
elems = args.elems  # None if not provided

st_time = time.time()
sim_setting = load_sim_settings(sim_file)
config_save(sim_file)


if elems is not None:
    elemSp(sim_setting, elems=elems, n_bootstrap=n_bootstrap)
else:
    elemSp(sim_setting, elems=["gc19_pc.cds"], n_bootstrap=n_bootstrap)
