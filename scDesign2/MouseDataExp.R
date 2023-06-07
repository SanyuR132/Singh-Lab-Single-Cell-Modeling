"
Desription:
  Experiment with scDesign2 method.

Author:
  Jiaqi Zhang
"

suppressPackageStartupMessages({
  library(scDesign2)
  library(Matrix)
})

source("./Models/R_Utils.R")
source("./Models/scDesign2/Running.R")


normalAugmentation <- function() {
  # Load mouse cell data
  filename <- "./Data/mouse_cell/wo_preprocess/training_all.mtx"
  data <- loadData(filename)
  # Fitting scDesign2
  print("Model fitting...")
  model <- modelFitting(data, sim_method = 'copula')
  # Make predictions
  test_filename <- "./Data/mouse_cell/wo_preprocess/testing_all.mtx"
  test_mtx_data <- loadData(test_filename)
  n_cell_new <- dim(test_mtx_data)[2]
  print("Model predicting...")
  predictions <- modelPrediction(model, n_cell_new, sim_method = 'copula')
  # Save prediction
  print("Prediction saving...")
  saveRDS(model, file = './Prediction/scDesign2/mouse_cell-scDesign2_copula_model.rds')
  writeMM(as(predictions, "sparseMatrix"), file = './Prediction/scDesign2/mouse_cell-scDesign2_estimation.mtx')
  print("Finished saving.")
}


#clusterAugmentation <- function() {
#  # Read data of cluster 3
#  print(sprintf("Read data of cluster 1..."))
#  filename <- "./Data/splat_simulation/wo_preprocess/cluster1_data-5008.mtx"
#  data <- loadData(filename) # genes x cells
#  # Different sampling ratio of training data
#  for (s in c(0.5, 0.25, 0.1, 0.05, 0.03, 0.01)) {
#    print(paste(replicate(70, '='), collapse = ""))
#    print(sprintf("Train data size : %f", s))
#    all_cells <- dim(data)[2]
#    ind_list <- c(1:all_cells)
#    # repeat 5 trials
#    for (t in 1:5) {
#      print(paste(replicate(70, '-'), collapse = ""))
#      print(sprintf("Trial %d", t))
#      # Split data into two sets
#      training_ind <- sample(ind_list, as.integer(all_cells * s), replace = FALSE)
#      removed_ind <- c(1:length(ind_list[-training_ind]))
#      train_data <- data[, training_ind]
#      removed_data <- data[, removed_ind]
#      # Fitting scDesign2
#      print("Model fitting...")
#      model <- modelFitting(train_data, sim_method = 'copula')
#      # Make predictions
#      n_cell_new <- dim(removed_data)[2]
#      print("Model predicting...")
#      predictions <- modelPrediction(model, n_cell_new, sim_method = 'copula')
#      # Save prediction
#      print("Prediction saving...")
#
#      saveRDS(model, file = sprintf('~/Projects/Structure_VAE_scRNA_Simulator/Prediction/scDesign2/splat_simulation_cluster1-trial%d-scDesign2_copula_model-%f.rds', t, s))
#      writeMM(as(predictions, "sparseMatrix"), file = sprintf('~/Projects/Structure_VAE_scRNA_Simulator/Prediction/scDesign2/splat_simulation_cluster1-trial%d-scDesign2_generated_data-%f.mtx', t, s))
#      writeMM(as(removed_data, "sparseMatrix"), file = sprintf('~/Projects/Structure_VAE_scRNA_Simulator/Prediction/scDesign2/splat_simulation_cluster1-trial%d-scDesign2_removed_data-%f.mtx', t, s))
#      writeMM(as(train_data, "sparseMatrix"), file = sprintf('~/Projects/Structure_VAE_scRNA_Simulator/Prediction/scDesign2/splat_simulation_cluster1-trial%d-scDesign2_train_data-%f.mtx', t, s))
#
#      #saveRDS(model, file = sprintf('./Prediction/scDesign2/PBMC_cluster3-trial%d-scDesign2_copula_model-%f.rds', t, s))
#      #writeMM(as(predictions, "sparseMatrix"), file = sprintf('./Prediction/scDesign2/PBMC_cluster3-trial%d-scDesign2_generated_data-%f.mtx', t, s))
#      #writeMM(as(removed_data, "sparseMatrix"), file = sprintf('./Prediction/scDesign2/PBMC_cluster3-trial%d-scDesign2_removed_data-%f.mtx', t, s))
#      #writeMM(as(train_data, "sparseMatrix"), file = sprintf('./Prediction/scDesign2/PBMC_cluster3-trial%d-scDesign2_train_data-%f.mtx', t, s))
#      print("Finished saving.")
#    }
#  }
#}

# --------------------------------------

normalAugmentation()
#clusterAugmentation()
