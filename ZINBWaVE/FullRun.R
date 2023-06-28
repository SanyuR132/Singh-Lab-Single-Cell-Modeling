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

library(argparse)
library(ggplot2)

source("./Running.R")
source("../R_Utils.R")


normalAugmentation <- function() {
  # Read data

  # save_dir = paste("/users/srajakum/scratch/Structure_VAE_scRNA_Simulator/baseline_results/ZINBWaVE/", args$study, "/tp_", args$ttp, sep='')
  date_time = format(Sys.time(), "%Y-%m-%d-at-%H-%M-%S")


  if (args$load_model_date_time != '') {
    save_dir = paste(save_dir, paste(load_model_date_time, "_ttp_", args$ttp, sep=''), sep='/')

    zinb_model = readRDS(paste(save_dir, 'WOT-ZINBWaVE_model.rds', sep = '/'))
    filteredTestData = loadData(paste(save_dir, 'WOT-ZINBWaVE_test_data.mtx', sep = '/'))
    print(sprintf('filtered test data shape: %s x %s', dim(filteredTestData)[1], dim(filteredTestData)[2]))
  } else {
    if (grepl("mouse", args$study)) {
      train_filename <- paste(data_dir,"mouse_cortex", args$study, paste("upto_tp", args$ttp, sep=''), "train_data.mtx", sep='/')
      print(train_filename)
      test_filename <- paste(data_dir, "mouse_cortex", args$study, paste("upto_tp", args$ttp, sep=''), "test_data.mtx", sep='/')
      print(test_filename)
    } else {
      train_filename <- paste(data_dir, args$study, paste("upto_tp", args$ttp, sep=''), "train_data.mtx", sep='/')
      test_filename <- paste(data_dir, args$study, paste("upto_tp", args$ttp, sep=''), "test_data.mtx", sep='/')
    }

    # create new timestamped directory
    save_dir = paste(save_dir, paste(date_time, "_ttp_", args$ttp, sep=''), sep='/')
    if (!dir.exists(save_dir)) {dir.create(save_dir)}
     
    trainData <- loadCellByGeneData(train_filename) # gene x cell
    trainData <- round(trainData) # gene x cell
    testData <- loadCellByGeneData(test_filename) # gene x cell

    print('Before filtering: ')
    print(paste("training data dim", c(dim(trainData)[1], dim(trainData)[2])))
    print(paste("testing data dim", c(dim(testData)[1], dim(testData)[2])))

    trainGeneFilter = which(rowSums(trainData) > 0)
    testGeneFilter = which(rowSums(testData) > 0)
    totalGeneFilter = intersect(trainGeneFilter, testGeneFilter)
    print(sprintf("Number of genes kept only in training data: %s", length(trainGeneFilter)))
    print(sprintf("Number of genes kept only in testing data: %s", length(testGeneFilter)))
    print(sprintf("Number of genes kept overall: %s", length(totalGeneFilter)))

    filteredTrainData = trainData[totalGeneFilter,]
    filteredTestData = testData[totalGeneFilter,]

    trainCellFilter = which(colSums(filteredTrainData) > 0)
    testCellFilter = which(colSums(filteredTestData) > 0)
    print(sprintf("Number of cells kept only in training data: %s", length(trainCellFilter)))
    print(sprintf("Number of cells kept only in testing data: %s", length(testCellFilter)))

    filteredTrainData = filteredTrainData[,trainCellFilter]
    filteredTestData = filteredTestData[,testCellFilter]

    print("remaining zero-count cells in train and test: ")
    print(which(colSums(filteredTrainData) == 0))
    print(which(colSums(filteredTestData) == 0))

    print("remaining zero-count genes in train and test: ")
    print(which(rowSums(filteredTrainData) == 0))
    print(which(rowSums(filteredTestData) == 0))

    print("After filtering :")
    print(sprintf('train data shape: %s x %s', dim(filteredTrainData)[1], dim(filteredTrainData)[2]))
    print(sprintf('test data shape: %s x %s', dim(filteredTestData)[1], dim(filteredTestData)[2]))

    print('Warnings in training: ')
    zinb_model = withCallingHandlers({  
      modelFitting(filteredTrainData)
      }, warning = function(w) {
        print(w)
        invokeRestart('muffleWarning')
      })
    
    saveRDS(zinb_model, file = paste(save_dir, paste(args$study, "-ZINBWaVE_model.rds", sep=''), sep = '/'))
    writeMM(as(filteredTestData, "sparseMatrix"), file = paste(save_dir, paste(args$study, "-ZINBWaVE_test_data.mtx", sep=''), sep = '/'))
    write.csv(totalGeneFilter, paste(save_dir, paste(args$study, "-ZINBWaVE_gene_filter.csv", sep=''), sep='/'), row.names = FALSE)
    print("Finished fitting and saving") 
  }

  n_cell_new = dim(filteredTestData)[2]

  print('Warnings in predicting: ')
  predictions = withCallingHandlers({
    predictions <- modelPrediction(zinb_model, num_predictions = n_cell_new)
    }, warning = function(w) {
      print(w)
      invokeRestart('muffleWarning')
    })
  print('Finished predicting')
  print(sprintf("predictions shape : %s x %s", dim(predictions)[1], dim(predictions)[2]))


  test_data = filteredTestData

  # plot histograms from each stat and get ks values
  stat_vec = c('mean_gene_ks', 'var_gene_ks', 'mean_var_gene_ks', 'mean_cell_ks', 'var_cell_ks', 'mean_var_cell_ks')
  ks_vec = c()
  hist_vec = c()
  num_bins = 50
  par(mfrow=c(2,3))
  for (stat_type in stat_vec) {
    out = get_ks(stat_type, test_data, predictions)
    ks_vec = c(ks_vec, out[1])
    labels = out[2]
    preds = out[3] 

    max_bin_val = ceiling(max(labels, preds))
    min_bin_val = floor(min(labels, preds))
    bins = pretty(min_bin_val:max_bin_val, n=num_bins)

    labels_hist = hist(labels, breaks=bins, freq=FALSE)
    preds_hist = hist(preds, breaks=bins, freq=FALSE)


    plot(labels_hist)
    plot(preds_hist, add=TRUE)
  }

  stat_df = data.frame('value' = ks_vec, row.names = stat_vec)
  write.csv(stat_df, file=paste(save_dir, "all_stats.csv", sep='/'))

  # ks_plots = ggarrange(hist_vec[1], hist_vec[2], hist_vec[3], hist_vec[4], hist_vec[5], hist_vec[6], labels = rownames(ks_df), ncol = 3, nrow=2)
  # ggsave(paste(save_dir, 'ks_plots.png', sep='/'))

  # compute Pearson and Spearman correlations
  mean_gene_labels = rowMeans(test_data)
  mean_gene_predictions = rowMeans(predictions)
  pcc = cor(mean_gene_labels, mean_gene_predictions, method = 'pearson')
  scc = cor(mean_gene_labels, mean_gene_predictions, method = 'spearman')
  stat_df = rbind(stat_df, data.frame('value'=c(pcc, scc), row.names=c('PCC', 'SCC')))
  ## cc_plot is unnecessary because you just get a straight a horizontal line (high correlation)
  # cc_plot = plot(mean_gene_labels, mean_gene_predictions, xlab='mean gene labels', ylab='mean gene predictions', col='blue')
  # text(40, 10, paste('PCC = ', round(pcc, 2) , "\n", "SCC = ", round(scc,2)))


  # no longer saving predictions since they take up too much space    
  # writeMM(as(predictions, "sparseMatrix"), file = paste(save_dir, 'WOT-ZINBWaVE_estimation.mtx', sep = '/'))
  print("Finished saving.")
}


get_ks <- function(stat_type, test_data, predictions) {
  eps = 1e-6
  if (stat_type == 'mean_cell_ks'){
    labels = mean_cell_labels = colMeans(test_data)
    predictions = mean_cell_predictions = colMeans(predictions)
  }
  if (stat_type == 'mean_gene_ks') {
    labels = mean_gene_labels = rowMeans(test_data)
    predictions = mean_gene_predictions = rowMeans(predictions)
  }

  if (stat_type == 'var_cell_ks'){  
    labels = var_cell_labels = mapply(var, asplit(test_data, 2))
    predictions = var_cell_predictions = mapply(var, asplit(predictions, 2))
  }

  if (stat_type == 'var_gene_ks')  {
    labels = var_gene_labels = mapply(var, asplit(test_data, 1))
    predictions = var_gene_predictions = mapply(var, asplit(predictions, 1))
  }

  if (stat_type == 'mean_var_cell_ks') {
    labels = mean_var_cell_labels = colMeans(test_data) / (mapply(var, asplit(test_data, 2)) + eps)
    predictions = mean_var_cell_predictions = colMeans(predictions) / (mapply(var, asplit(predictions, 2)) + eps)
  }

  if (stat_type == 'mean_var_gene_ks') {
    labels = mean_var_gene_labels = rowMeans(test_data) / (mapply(var, asplit(test_data, 1)) + eps)
    predictions = mean_var_gene_predictions = rowMeans(predictions) / (mapply(var, asplit(predictions, 1)) + eps)
  }

  ks = ks.test(predictions, labels)$statistic

  return (c(ks, labels, predictions))

}

#clusterAugmentation <- function() {
#  # Read data of cluster 3
#  print(sprintf("Read data of cluster 1..."))
#  filename <- "./Data/splat_simulation/wo_preprocess/cluster1_data-5008.mtx"
#  data <- loadData(filename) # genes x cells
#  data <- round(data)
#  gene_filer <- filterGene(data)
#  filtered_data <- gene_filer$data
#  filter <- gene_filer$filter
#  shape <- dim(filtered_data)
#  n_cells <- shape[2]
#  n_genes <- shape[1]
#  print("After filtering :")
#  print(sprintf("Num of cells : %s", n_cells))
#  print(sprintf("Num of genes : %s", n_genes))
#  write.csv(filter, './Prediction/ZINBWaVE/splat_simulation_cluster1-ZINBWaVE-filter.csv')
#  # Different sampling ratio of training data
#  for (s in c(0.5, 0.25, 0.1, 0.05, 0.03, 0.01)) {
#    print(paste(replicate(70, '='), collapse = ""))
#    print(sprintf("Train data size : %f", s))
#    all_cells <- dim(filtered_data)[2]
#    ind_list <- c(1:all_cells)
#    # repeat 5 trials
#    for (t in 1:5) {
#      print(paste(replicate(70, '-'), collapse = ""))
#      print(sprintf("Trial %d", t))
#      # Split data into two sets
#      training_ind <- sample(ind_list, as.integer(all_cells * s), replace = FALSE)
#      removed_ind <- c(1:length(ind_list[-training_ind]))
#      train_data <- filtered_data[, training_ind]
#      removed_data <- filtered_data[, removed_ind]
#      # fitting ZINB-Wave model
#      zinb_model <- modelFitting(train_data)
#      print("Finished fitting...")
#      # simulate with ZINB-WaVe
#      n_cell_new <- dim(removed_data)[2]
#      predictions <- modelPrediction(zinb_model, num_predictions = n_cell_new)
#      print("Finished simulation...")
#      # Save data
#      print("Prediction saving...")
#      saveRDS(model, file = sprintf('./Prediction/ZINBWaVE/splat_simulation_cluster1-trial%d-ZINBWaVE_copula_model-%f.rds', t, s))
#      writeMM(as(predictions, "sparseMatrix"), file = sprintf('./Prediction/ZINBWaVE/splat_simulation_cluster1-trial%d-ZINBWaVE_generated_data-%f.mtx', t, s))
#      writeMM(as(removed_data, "sparseMatrix"), file = sprintf('./Prediction/ZINBWaVE/splat_simulation_cluster1-trial%d-ZINBWaVE_removed_data-%f.mtx', t, s))
#      writeMM(as(train_data, "sparseMatrix"), file = sprintf('./Prediction/ZINBWaVE/splat_simulation_cluster1-trial%d-ZINBWaVE_train_data-%f.mtx', t, s))
#      print("Finished saving.")
#    }
#  }
#}

# --------------------------------------

parser <- ArgumentParser()
parser$add_argument('--load_model_date_time', type = "character", default = '')
parser$add_argument('-ttp', type = "character")
parser$add_argument('--study', type = "character")
parser$add_argument('--save_dir', type = "character")
parser$add_argument('--data_dir', type = "character")
args <- parser$parse_args()

save_dir = args$save_dir
data_dir = args$data_dir
load_model_date_time = args$load_model_date_time
ttp = args$ttp
study = args$study

normalAugmentation()
#clusterAugmentation()
