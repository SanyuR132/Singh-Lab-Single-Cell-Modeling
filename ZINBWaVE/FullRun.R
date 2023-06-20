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

source("./Running.R")
source("../R_Utils.R")


normalAugmentation <- function() {
  # Read data

  # save_dir = paste("/users/srajakum/scratch/Structure_VAE_scRNA_Simulator/baseline_results/ZINBWaVE/", args$study, "/tp_", args$ttp, sep='')
  if (!dir.exists(save_dir)) {dir.create(save_dir)}

  if (args$load_model) {
    zinb_model = readRDS(paste(save_dir, 'WOT-scDesign2_copula_model.rds', sep = '/'))
    filteredTestData = loadCellByGeneData(paste(save_dir, 'WOT-ZINBWaVE_test_data.mtx', sep = '/'))
    print(sprintf('filtered test data shape: %s x %s', dim(filteredTestData)[1], dim(filteredTestData)[2]))
  } else {
    if (grepl("mouse", args$study)) {
      train_filename <- paste(data_dir,"mouse_cortex", args$study, "upto_tp", args$ttp, "train_data.mtx", sep='/')
      print(train_filename)
      test_filename <- paste(data_dir, "mouse_cortex", args$study, "upto_tp", args$ttp, "test_data.mtx", sep='/')
      print(test_filename)
    } else {
      train_filename <- paste(data_dir, args$study, "upto_tp", args$ttp, "train_data.mtx", sep='/')
      test_filename <- paste(data_dir, args$study, "upto_tp", args$ttp, "test_data.mtx", sep='/')
    }
     
    trainData <- loadCellByGeneData(train_filename) # gene x cell
    trainData <- round(trainData) # gene x cell
    testData <- loadCellByGeneData(test_filename) # gene x cell

    trainGeneFilter = which(rowSums(trainData) > 0)
    testGeneFilter = which(rowSums(testData) > 0)
    totalGeneFilter = intersect(trainGeneFilter, testGeneFilter)
    print(sprintf("Number of genes kept only in training data: %s", length(trainGeneFilter)))
    print(sprintf("Number of genes kept only in testing data: %s", length(testGeneFilter)))
    print(sprintf("Number of genes kept overall: %s", length(totalGeneFilter)))

    print('Before filtering: ')
    print(paste("training data dim", c(dim(trainData)[1], dim(trainData)[2])))
    print(paste("testing data dim", c(dim(testData)[1], dim(testData)[2])))

    filteredTrainData = trainData[totalGeneFilter,]
    filteredTestData = testData[totalGeneFilter,]

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
    
    saveRDS(zinb_model, file = paste(save_dir, "WOT-ZINBWaVE_model.rds", sep = '/'))
    writeMM(as(filteredTestData, "sparseMatrix"), file = paste(save_dir, 'WOT-ZINBWaVE_test_data.mtx', sep = '/'))
    write.csv(totalGeneFilter, paste(save_dir, "WOT-ZINBWaVE_gene_filter.csv", sep='/'), row.names = FALSE)
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

  # compute marginal KS statistics

  test_data = filteredTestData

  mean_cell_labels = colMeans(test_data)
  mean_cell_predictions = colMeans(predictions)
  mean_cell_ks = ks.test(mean_cell_predictions, mean_cell_labels)$statistic

  mean_gene_labels = rowMeans(test_data)
  mean_gene_predictions = rowMeans(predictions)
  mean_gene_ks = ks.test(mean_gene_predictions, mean_gene_labels)$statistic

  var_cell_labels = mapply(var, asplit(test_data, 2))
  var_cell_predictions = mapply(var, asplit(predictions, 2))
  var_cell_ks = ks.test(var_cell_predictions, var_cell_labels)$statistic

  var_gene_labels = mapply(var, asplit(test_data, 1))
  var_gene_predictions = mapply(var, asplit(predictions, 1))
  var_gene_ks = ks.test(var_gene_predictions, var_gene_labels)$statistic

  eps = 1e-6
  mean_var_cell_labels = mean_cell_labels / (var_cell_labels + eps)
  mean_var_cell_predictions = mean_cell_predictions / (var_cell_predictions + eps)
  mean_var_cell_ks = ks.test(mean_var_cell_predictions, mean_var_cell_labels)$statistic

  mean_var_gene_labels = mean_gene_labels / (var_gene_labels + eps)
  mean_var_gene_predictions = mean_gene_predictions / (var_gene_predictions + eps)
  mean_var_gene_ks = ks.test(mean_var_gene_predictions, mean_var_gene_labels)$statistic

  ks_df = data.frame('ks' = c(mean_gene_ks, var_gene_ks, mean_var_gene_ks, mean_cell_ks, var_cell_ks, mean_var_cell_ks), row.names = c('mean_gene', 'var_gene', 'mean_var_gene', 'mean_cell', 'var_cell', 'mean_var_cell'))
  write.csv(ks_df, file=paste(save_dir, "ks_stats.csv", sep='/'))

  # plot histograms from each stat
  num_bins = 50
  for (stat_type in rownames(ks_df)) {
    labels = paste(stat_type, '_labels', sep='')
    preds = paste(stat_type, '_preds', sep='')

    max_bin_val = max(max(labels, preds))
    min_bin_val = min(min(labels, preds))
    bins = pretty(min_bin_val:max_bin_val, n=num_bins)

    labels_hist = hist(labels, breaks=bins, freq=FALSE)
    preds_hist = hist(l, breaks=bins, freq=FALSE)

    as.name(paste(stat_type, "_hist", sep='')) = plot(labels_hist)
    plot(preds_hist, add=TRUE)
  }

  ks_plots = ggarange(mean_gene_hist, var_gene_hist, mean_var_gene_hist, mean_cell_hist, var_cell_hist, mean_var_cell_hist, labels = rownames(ks_df), ncol = 3, nrow=2)
  ggsave(paste(save_dir, 'ks_plots.png', sep='/'))

  # compute Pearson and Spearman correlations
  pcc = cor(mean_gene_labels, mean_gene_predictions, method = 'pearson')
  scc = cor(mean_gene_labels, mean_gene_predictions, method = 'spearman')
  cc_plot = plot(mean_gene_labels, mean_gene_predictions, xlab='mean gene labels', ylab='mean gene predictions', col='blue')
  text(0.65, 0.25, paste('PCC = ', pcc , "\n", "SCC = ", scc))

  ggsave(paste(save_dir, 'cc_plot.png', sep='/'))

  # no longer saving predictions since they take up too much space    
  # writeMM(as(predictions, "sparseMatrix"), file = paste(save_dir, 'WOT-ZINBWaVE_estimation.mtx', sep = '/'))
  print("Finished saving.")

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
parser$add_argument('--load_model', type = "logical", default = FALSE)
parser$add_argument('-ttp', type = "character")
parser$add_argument('--study', type = "character")
parser$add_argument('--save_dir', type = "character")
parser$add_argument('--data_dir', type = "character")
args <- parser$parse_args()

save_dir = args$save_dir

normalAugmentation()
#clusterAugmentation()
