rm(list = ls())

# Load necessary libraries
library(data.table)
library(dplyr)

# Define the main directory containing the cancer type subdirectories
main_dir <- "/home/bahari/iDriver/extdata/procInput/BMRs/observed/"

# Get the list of cancer type subdirectories
cancer_dirs <- list.dirs(main_dir, recursive = FALSE)

# Initialize an empty list to store data frames for each cancer type
all_datas <- list()

# Loop over each cancer type directory
for (cancer_dir in cancer_dirs) {
  
  # Extract the cancer type from the directory path
  cancer_type <- basename(cancer_dir)
  # print(cancer_type)
  
  
  all_data <- list()
  # Get the list of assessment files in the current directory
  assessment_files <- list.files(cancer_dir, pattern = "_assessment", full.names = TRUE, recursive = TRUE)
  
  # Read and process each assessment file
  for (file in assessment_files) {
    # Read the data from the file
    data <- fread(file)
    
    # Extract the 'corr' row
    corr_row <- data[grepl("corr_", V1)]
    
    
    
    # Reshape the data to have one row per element type
    melted_data <- melt(corr_row, id.vars = "V1", variable.name = "element_type", value.name = "corr")
    
    # Add the cancer type column
    melted_data[, cancer_type := cancer_type]
    
    # Add the processed data to the list
    all_data[[length(all_data) + 1]] <- melted_data
  }
  
  # Check if both eMET and GBM data are available
  if (length(all_data) == 2) {
    df <- left_join(all_data[[1]], all_data[[2]], by = c('element_type', 'cancer_type'))
    df <- df[, c('element_type', 'corr.x', 'cancer_type', 'corr.y')]
    colnames(df) <- c('element_type', 'corr_eMET', 'cancer_type', 'corr_GBM')
    df <- df[, c('element_type', 'cancer_type', 'corr_GBM', 'corr_eMET')]
    
    # Append the dataframe to the list
    all_datas <- append(all_datas, list(df))
  } else if (length(all_data) == 0) {
    # Print a message if the cohort does not have both correlations
    cat(cancer_type, "doesn't have both of intergenic and eMET corrs\n")
  }
}

# Combine all data frames into one
final_data <- rbindlist(all_datas)

##########################################################################################################

rm(list = ls())

library(data.table)
library(ggplot2)
library(reshape2)
library(fmsb)


data <- fread('../../tmp_corr_eMET_GBM_allCohorts.tsv')


# Get unique cancer types
cancer_types <- unique(data$cancer_type)

# Create radar plots for each cancer type
for (cancer in cancer_types) {
  
  # Filter data for the current cancer type
  filtered_data <- subset(data, cancer_type == cancer)
  
  filtered_data <- filtered_data[,c('element_type', 'corr_GBM', 'corr_eMET')]
  radar_data <- t(filtered_data)
  element_names <- c('enhancers' = 'Enhancers',
                     'gc19_pc.3utr' = '3\' UTR',
                     'gc19_pc.5utr' = '5\' UTR',
                     'gc19_pc.cds' = 'CDS',
                     'gc19_pc.promCore' = 'Core Promoter',
                     'gc19_pc.ss' = 'Splice site'
                     # , 'lncrna.ncrna' = 'lncRNA',
                     # 'lncrna.promCore' = 'lncRNA Promoter'
  )
  colnames(radar_data) <- radar_data[1,] #element_names[radar_data[1,]]
  radar_data <- radar_data[2:3,]
  minimum = min(as.numeric(radar_data) - .1)
  
  # Set max and min for radar chart
  radar_data <- rbind(max = 1, min = minimum, radar_data)
  radar_data <- as.data.frame(radar_data)
  radar_data <- data.frame(lapply(radar_data, function(x) as.numeric(as.character(x))))
  
 
 
  
  # Plot radar chart
  
  radarchart(radar_data, axistype = 1,
             pcol = c('#386cb0', '#e41a1c'),
             pfcol = c(scales::alpha('#386cb0', 0.3), scales::alpha('#e41a1c', 0.1)),
             cglcol="grey", cglty=1, axislabcol="grey", caxislabels='', cglwd=0.8,
             plwd = 2,
             title = cancer)
  legend(x = "topright", legend = c("Intergenic", "eMET"), col = c( '#386cb0', '#e41a1c'), lwd = 2)
  
}

