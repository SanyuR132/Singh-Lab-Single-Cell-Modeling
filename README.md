# Structure VAE Model for scRNA-seq Data

This repository contains the code for running, tuning, and analysing `ZINB-WaVE` and `scVAE`, two of the baseline models for dimensionality reduction of scRNA-seq data

------------  

## Quick Start

These instructions are specific to Oscar. For running locally, one can use the commands contained in the bash scripts without all the Oscar-specific specifications (i.e, you can use the uncommented stuff).

### ZINB-WaVE
* Run `module load R/{version}` in the command line with the specific version of R in which the necessary packages are installed (see "Models")
* Navigate to `ZINBWaVE` directory
* There is a different script to run the model on each dataset. For example, the script to run it on the drosophila data is `run_dros.sh`. These scripts run `FullRun.R`

You may need to edit the script to specify arguments or change Oscar specifications. Do this with `vim {file_name}`
The command line arguments are as follows:
* `--load_model_date_time`: time stamp of directory that contains the saved model; use this if you have a saved model from a previous run (or a run where an error occurs after training is finished) as avoiding retraining will save time
* `-ttp`: abbreviation of truncate time point; used to specify upto what timpepoint data should be loaded; set this to 0 for non-time series datasets
* `--study`: specify what dataset to use; 'drosophila', 'WOT', 'PBMC', 'mouse_cortex1_chromium', 'mouse_cortex2_chromium', 'mouse_cortex1_smart_seq', 'mouse_cortex2_smart_seq'
* `--save_dir`: path to directory where outputs should be saved
* `--data_dir`: path to directory where all the data are stored (not to the data from the specific study, but the directory above that)

A new timestamped directory will be created inside the specified save directory with the following format: {year}-{month}-{day}-at-{hour}-{minute}-{second}
This directory will contain the following outputs:
* `all_stats.csv`: a csv file with all the marginal ks statistics as well the pearson and spearman correlations

Tuning is not implemented for ZINB-WaVE because R does not have a satisfactory tuning package (I tried `ParBayesianOptimization` but it had a lot of issues)

### scVAE
* Navigate to `scvae` directory
* There is a different script to run/tune the model on each dataset. `run_{dataset}.sh` runs `FullRun.py`, and `run_{dataset}_tuning.sh` runs `tune.py`.
* Generally, you should first tune the model on a given dataset and then use the tuned hyperparameters to do run the model

You may need to edit the script to specify arguments or change Oscar specifications. Do this with `vim {file_name}`
The command line arguments for `FullRun.py` are as follows:
* `-lhsdt`, `--load_hyper_struct_date_and_time`: the timestamp ({year}-{month}-{day}-at-{hour}-{minute}-{second}) of the directory in which the tuned hyperparameters are stored
* `-ttp`, `--truncate_time_point`: used to specify upto what timpepoint data should be loaded; set this to 0 for non-time series datasets
* `-stu`, `--study`: specify what dataset to use; 'drosophila', 'WOT', 'PBMC', 'mouse_cortex1_chromium', 'mouse_cortex2_chromium', 'mouse_cortex1_smart_seq', 'mouse_cortex2_smart_seq'
* `--save_dir`: path to directory where outputs should be saved
* `--data_dir`: path to directory where all the data are stored (not to the data from the specific study, but the directory above that)

A new timestamped directory will be created inside the specified save directory with the following format: {year}-{month}-{day}-at-{hour}-{minute}-{second}_tuned ('_tuned' is appended if used loaded hyperparameters)
This directory will contain the following outputs:
* `ks_stats.csv`: a csv file with all the marginal ks statistics
* `ks_plots.png`: histograms of all the marginal distributions (true and predicted on the same graph)
* `cc_stats.csv`: a csv file with all the Pearson and Spearman correlations
* `cc_plot.png`: a scatter plot of the per-gene mean values of the true vs predicted expression matrix
* `info.txt`: text file with basic information of model run (date and time, hyperparameters)

The command line arguments for `tune.py` are as follows:
* `-ttp`, `--truncate_time_point`: used to specify upto what timpepoint data should be loaded; set this to 0 for non-time series datasets
* `-stu`, `--study`: specify what dataset to use; 'drosophila', 'WOT', 'PBMC', 'mouse_cortex1_chromium', 'mouse_cortex2_chromium', 'mouse_cortex1_smart_seq', 'mouse_cortex2_smart_seq'
* `-n`, `--num_trials`: number of tuning trials (you should prefer to use tuning time so that you can preciesly set the walltime in Oscar)
* `-tuti`, `--tuning_time`: number of minutes for tuning
* `--tune_all`: this does not require input, simply including it in the command line sets it to true; keep this on by default
* arguments for choosing which parameters to tune (if `--tune_all`) is not activated. See scVAE documentation (linked in "Models" section)
* `--save_dir`: path to directory where outputs should be saved
* `--data_dir`: path to directory where all the data are stored (not to the data from the specific study, but the directory above that)

A new timestamped directory will be created inside the specified save directory with the following format: {year}-{month}-{day}-at-{hour}-{minute}-{second}
This directory will contain the following outputs:
* `param_imp.png`: bar plot of 'param importances'. See [documentation](https://optuna.readthedocs.io/en/stable/reference/visualization/generated/optuna.visualization.plot_param_importances.html)
* `opt_hist.png`: plot of trial number against objective value (KS value); inspect this to gauge if model was tuned for long enough 
* `tuning_info.txt`: basic info and tuned hyperparameter values
* `hyperparameters.pt`: pickled dictionary of hyperparameter values

-----------

## Data
Four datasets are used: two whiche are time series and two which are not

### non-time series  
* [Mouse Cortex](https://singlecell.broadinstitute.org/single_cell/study/SCP425/single-cell-comparison-cortex-data) dataset
    * Consists of two experiment (Cortex1 and Cortex2)
    * 4 different single-nuclear RNA-seq methods per experiment – we only use two methods (10x Chromium and Smart-Seq2)
    * The Smart-Seq2 data seem to have very few cells, so use those with caution
* [PBMC 68K](https://www.10xgenomics.com/resources/datasets):  
    * The fresh 68k PBMCs (Donor A) dataset 
    * Data can be dowloaded from this [link](http://cf.10xgenomics.com/samples/cell-exp/1.1.0/fresh_68k_pbmc_donor_a/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz)

### time-series
* [Drosophila](https://www.science.org/doi/10.1126/science.abn5800?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200pubmed) dataset
    * Consists of 10 timepoints 
    * The number of cells is quite large (~185,000), so we subsample to 5000 cells per timepoint (50,000 total)
    * 2000 top highly variable genes are selected
* [WOT](https://doi.org/10.1016/j.cell.2019.01.006)(**W**addington **O**ptimal-**T**ransport)
    * Consists of 16 timepoints
    * ~1400 genes are pre-selected (preprocessing done by authors)

------------

## Models

* [ZINB-WaVE](https://www.nature.com/articles/s41467-017-02554-5): ZINB-Wave is designed to learn low-dimensional 
representation of single-cell data, by modeling data as zero-inflated negative binomial variables. 
We install the zinbwave [R package](https://github.com/drisso/zinbwave). Notice that this method is not designed for data 
simulation, but it provides such functions. We adapted scripts from a 
[benchmark analysis](https://github.com/HelenaLC/simulation-comparison/tree/master/code) for using ZINB-WaVE.
Requirements: zinbwave, Matrix, argparse

* [scVAE](https://academic.oup.com/bioinformatics/article/36/16/4415/5838187): We download source codes from the 
[GitHub repository](https://github.com/scvae/scvae) and invoke functions in [Models/scvae/Running.py](Models/scvae/Running.py).
Requirements: python==3.6, tensorflow==1.15.2, torch==1.8.1, scanpy, imprtlib-resources, loompy, tensorflow-probability==0.7.0
