<p align="left">
  <img src="eMETLogo.png" alt="Project Logo" width="200">
</p>

# eMET


**e**lement-specific **M**utation **E**stimator with boosted **T**rees

eMET is a tool for building an accurate background mutation rate (BMR) model to help identify cancer drivers in coding and non-coding regions. It uses boosted trees to leverage extensive intergenic data, from somatic point mutations in a tumor cohort, and fine-tunes the model using element-specific information. The process involves building an initial model with intergenic data across a comprehensive set of (epi)genomic features, followed by enhancement through bootstrap samples incorporating element-specific data. Genomic elements with higher-than-expected mutation recurrence than the background suggest positive selection in the cohort and are introduced as candidate driver elements.

<p align="center">
  <img src="pipeline.jpg" alt="Pipeline Overview" width="800">
</p>


## How to Use eMET

### 1. Installation

```bash
git clone https://github.com/FaridehBahari/eMET.git 
```

### 2. Prepare Inputs

#### a. Genomic Intervals

- **Restrict the functional element coordinates to callable regions of the genome:**

```bash
bedtools intersect -a <path/to/functional_elements.bed6> -b <path/to/callable.bed.gz> > <path/to/save/callable_functional_elements.bed6>
```

- **Make intergenic genomic intervals:**

Use any approach provided in prepareInputs.py to generate fixed-size or variable-size intergenic bins. Variable-size intergenic bins are preferred.

#### b.  Make response tables

Create response tables for both functional elements and intergenic intervals. These tables record the observed number of mutations in each genomic interval, the length of each interval (summing lengths if composed of several blocks), the number of donors in the cohort, and the number of mutated samples.
To create the response table, generate a GRanges object from the somatic mutations with at least one metadata column named D_id for donor IDs.

```bash
Rscript preprocess/save_responseTable.R <path/to/mutations_granges.RData> <path/to/intervals.bed6> <path/to/save> 'save_name' 
```

#### c. Generate feature matrix
 Use the [DriverPower](https://github.com/smshuai/DriverPower/tree/master/script/make_features) scripts to generate the feature matrix. It is recommended to save the matrix as a .h5 file.

### 3. Build background mutation rate (BMR) model:

#### a. Train the intergenic XGBoost model

```bash
python RUN_BMR.py configs/sim_setting_intergenic.ini
```

#### b. Fine-tune the intergenic model with transfer learning

```bash
python run_eMET.py configs/sim_setting_finetunning.ini  <path/to/intergenic/pretrainedModel.pkl'> <#bootstraps>
```

### 4. Run burden test

```bash
python run_burdenTest.py <path/to/BMR/directory>
```

## Reference
If you use eMET in your research, please cite:

Bahari, F., Ahangari Cohan, R., & Montazeri, H. (2025). 
Element-specific estimation of background mutation rates in whole cancer genomes through transfer learning.  
*NPJ Precision Oncology*. [https://doi.org/10.1038/s41698-025-00871-3](https://doi.org/10.1038/s41698-025-00871-3)


## Contact

For any questions or issues, please open an issue on this repository or contact [bahari.faride@gmail.com].
