import os
import argparse
from inference.infer import perform_burdenTest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


parser = argparse.ArgumentParser(description='Perform burden test for a specific BMR setting based on the path to the directory containing predictions')
parser.add_argument('dir_path', type=str, help='the path to the parent directory containing sim_setting and predictions')
args = parser.parse_args()
dir_path = args.dir_path
    
perform_burdenTest(dir_path)


################### Run the following lines of code for cohort-specific inference
# import os
# from inference.infer import perform_burdenTest

# # Suppress TensorFlow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # Define the base path where directories are located
# base_dir = '../external/BMR/output/reviewerComments/'
# cohorts = os.listdir(base_dir)

# # List all subdirectories in the base path
# dir_paths = [os.path.join(base_dir, subdir) for subdir in cohorts if os.path.isdir(os.path.join(base_dir, subdir))]

# # Iterate over each directory and apply perform_burdenTest for each cohort
# for cohort in cohorts:
#     dir_path = f'../external/BMR/output/reviewerComments/{cohort}/eMET/'
#     print(f'Processing cohort: {cohort} in directory: {dir_path}')
#     perform_burdenTest(dir_path, cohort)
