"
Desription:
  Experiment with ZINB-WaVE method.


Author:
  Jiaqi Zhang
"
#


#suppressPackageStartupMessages({
#  library(zinbwave)
#  library(Matrix)
#  library(BiocParallel)
#  library(SingleCellExperiment)
#})

source("./Models/ZINBWaVE/Running.R")
source("./Models/R_Utils.R")


normalAugmentation <- function() {
  # Read data
  filename <- "./Data/PBMC/wo_preprocess/training_data.mtx"
  data <- loadCellByGeneData(filename)
  #data <- data[, c(1:50)] # TODO: select only 50 samples for efficiency
  data <- round(data)
  n_cells <- dim(data)[2]
  gene_filer <- filterGene(data)
  filtered_data <- gene_filer$data
  filter <- gene_filer$filter
  shape <- dim(filtered_data)
  n_cells <- shape[2]
  n_genes <- shape[1]
  print("After filtering :")
  print(sprintf("Num of cells : %s", n_cells))
  print(sprintf("Num of genes : %s", n_genes))
  write.csv(filter, './Prediction/ZINBWaVE/PBMC-ZINBWaVE-filter.csv')
  # fitting ZINB-Wave model
  zinb_model <- modelFitting(filtered_data)
  print("Finished fitting...")
  # simulate with ZINB-WaVe
  test_filename <- "./Data/PBMC/wo_preprocess/testing_data.mtx"
  test_mtx_data <- loadCellByGeneData(test_filename)
  n_cell_new <- dim(test_mtx_data)[2]
  predictions <- modelPrediction(zinb_model, num_predictions = n_cell_new)
  print("Finished simulation...") # Save data
  saveRDS(zinb_model, file = './Prediction/ZINBWaVE/PBMC-ZINBWaVE_model.rds')
  writeMM(as(predictions, "sparseMatrix"), file = './Prediction/ZINBWaVE/PBMC-ZINBWaVE_esimation.mtx')
  print("Finished saving.")
}


clusterAugmentation <- function() {
  # Read data of cluster 3
  print(sprintf("Read data of cluster 3..."))
  filename <- "./Data/PBMC/clusters/cluster3_data-4574.mtx"
  data <- loadCellByGeneData(filename) # genes x cells
  data <- round(data)
  gene_filer <- filterGene(data)
  filtered_data <- gene_filer$data
  filter <- gene_filer$filter
  shape <- dim(filtered_data)
  n_cells <- shape[2]
  n_genes <- shape[1]
  print("After filtering :")
  print(sprintf("Num of cells : %s", n_cells))
  print(sprintf("Num of genes : %s", n_genes))
  write.csv(filter, './Prediction/ZINBWaVE/PBMC_cluster3-ZINBWaVE-filter.csv')
  # Different sampling ratio of training data
  for (s in c(0.5, 0.25, 0.1, 0.05, 0.03, 0.01)) {
    print(paste(replicate(70, '='), collapse = ""))
    print(sprintf("Train data size : %f", s))
    all_cells <- dim(filtered_data)[2]
    ind_list <- c(1:all_cells)
    # repeat 5 trials
    for (t in 1:5) {
      print(paste(replicate(70, '-'), collapse = ""))
      print(sprintf("Trial %d", t))
      # Split data into two sets
      training_ind <- sample(ind_list, as.integer(all_cells * s), replace = FALSE)
      removed_ind <- c(1:length(ind_list[-training_ind]))
      train_data <- filtered_data[, training_ind]
      removed_data <- filtered_data[, removed_ind]
      # fitting ZINB-Wave model
      zinb_model <- modelFitting(train_data)
      print("Finished fitting...")
      # simulate with ZINB-WaVe
      n_cell_new <- dim(removed_data)[2]
      predictions <- modelPrediction(zinb_model, num_predictions = n_cell_new)
      print("Finished simulation...")
      # Save data
      print("Prediction saving...")
      saveRDS(model, file = sprintf('./Prediction/ZINBWaVE/PBMC_cluster3-trial%d-ZINBWaVE_copula_model-%f.rds', t, s))
      writeMM(as(predictions, "sparseMatrix"), file = sprintf('./Prediction/ZINBWaVE/PBMC_cluster3-trial%d-ZINBWaVE_generated_data-%f.mtx', t, s))
      writeMM(as(removed_data, "sparseMatrix"), file = sprintf('./Prediction/ZINBWaVE/PBMC_cluster3-trial%d-ZINBWaVE_removed_data-%f.mtx', t, s))
      writeMM(as(train_data, "sparseMatrix"), file = sprintf('./Prediction/ZINBWaVE/PBMC_cluster3-trial%d-ZINBWaVE_train_data-%f.mtx', t, s))
      print("Finished saving.")
    }
  }
}

# --------------------------------------

normalAugmentation()
#clusterAugmentation()
