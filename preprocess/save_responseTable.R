rm(list=ls())

source("preprocess/make_responseTable.R")


# Capture command-line arguments
args <- commandArgs(trailingOnly = TRUE)

# Assign arguments to variables
path_to_gr <- args[1]
path_bed <- args[2]
path_out <- args[3]
save_name <- args[4]

save_responseTable(path_to_gr, path_bed, path_out, save_name)

print(' ************** Job Done **************')

