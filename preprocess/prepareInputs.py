######################### Prepare functional elements #######################
# bedtools intersect -a ../external/database/bins/raw/PCAWG_test_genomic_elements.bed6 -b ../external/database/bins/raw/callable.bed.gz > ../external/database/bins/proccessed/PCAWG_callable.bed6

######################### Generate variable-size intergenic bins ###################
# step1: remove pcawg elements from callable
# bedtools subtract  -a ../external/database/bins/raw/callable.bed.gz -b ../external/database/bins/raw/PCAWG_test_genomic_elements.bed6  > ../external/database/bins/proccessed/callable_withoutPCAWGs.bed


import re
import numpy as np
import pandas as pd

path_all_bins = '../external/database/bins/proccessed/callable_withoutPCAWGs.bed'
callables = pd.read_csv(path_all_bins, header=None, sep='\t', 
                        names=['chr', 'start', 'end'])

# Step2:filterout sex chromosomes and save the bins as bed6 file
filtered_df = callables[~callables['chr'].isin(['chrX', 'chrY'])]
filtered_df = filtered_df.sort_values(by=['chr', 'start'])
filtered_df['name'] = [f'v{i}' for i in (range(filtered_df.shape[0]))]
filtered_df['score'] = 0
filtered_df['strand'] = '+'
filtered_df.to_csv('../external/database/bins/proccessed/callable_intergenic_intervals_wo_pcawg.bed6',
                    sep = '\t', header=False, index=False)

######################### Generate fixed-size intergenic bins  ###################
# bedtools makewindows -g ../external/database/bins/raw/hg19_chromSize.txt  -w 100000 | awk 'BEGIN{OFS="\t"} $1 ~ /^chr[0-9]+$/ {print $1, $2, $3, "b"NR, "1000", "."}' > ../external/database/bins/proccessed_bedtools/100k_window.bed
# bedtools intersect -a ../external/database/bins/proccessed_bedtools/100k_window.bed -b ../external/database/bins/raw/callable.bed.gz > ../external/database/bins/proccessed_bedtools/callabel_100k_window.bed
# bedtools subtract -a ../external/database/bins/proccessed_bedtools/callabel_100k_window.bed -b ../external/database/bins/raw/PCAWG_test_genomic_elements.bed6 > ../external/database/bins/proccessed_bedtools/callable_100k_intergenic_bins.bed6