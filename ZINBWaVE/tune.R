"
Description:
  Automatically tune model parameters.

Author:
  Jiaqi Zhang
"

library("xgboost")
library("ParBayesianOptimization")
library("argparse")
library("dgof")
library("ggplot2")
library(Matrix)
library(BiocParallel)
library(zinbwave)

source("../R_Utils.R")
# source("./Running.R")


parser <- ArgumentParser()

parser$add_argument('--study', type = "character", help = 'name of data source {zebrafish, WOT, or new_mouse_cell}')
parser$add_argument('-n', "--num_trials", type = "integer", default = 200)
parser$add_argument('--max_latent_size', type = "integer", default = 50)
args <- parser$parse_args()

modelPrediction <- function (model, num_predictions) {
  print('Started Predicting')
  pred_res <- zinbSim(model)
  predictions <- pred_res$count
  while (num_predictions > dim(predictions)[2]){
    predictions <- cbind(predictions, zinbSim(model)$count)
  }
  ind_list <- c(1:dim(predictions)[2])
  rand_idx <- sample(ind_list, num_predictions, replace = FALSE)
  predictions <- predictions[, rand_idx]
  return (predictions)
}


objective <- function(latent_size) {

  stopifnot(class(latent_size) == 'numeric')

  model <- zinbFit(train_data, K = latent_size, verbose = FALSE, BPPARAM=MulticoreParam(parallel::detectCores())) 
  predictions <- modelPrediction(model, n_cell_new)

  stopifnot(dim(predictions) == dim(test_data))

  print('Finished model fitting and predicting')

  print(sprintf("predictions shape : %s x %s", dim(predictions)[1], dim(predictions)[2]))

  ## DEBUGGING: using chi squared test instead of ks. also not changing variable names 'cause meh 
  mean_cell_labels = colMeans(test_data)
  mean_cell_predictions = colMeans(predictions)
  # mean_cell_ks = ks.test(mean_cell_predictions, mean_cell_labels)$statistic
  mean_cell_ks = chisq.test(mean_cell_predictions, mean_cell_labels)$statistic

  mean_gene_labels = rowMeans(test_data)
  mean_gene_predictions = rowMeans(predictions)
  # mean_gene_ks = ks.test(mean_gene_predictions, mean_gene_labels)$statistic
  mean_gene_ks = chisq.test(mean_gene_predictions, mean_gene_labels)$statistic

  var_cell_labels = mapply(var, asplit(test_data, 2))
  var_cell_predictions = mapply(var, asplit(predictions, 2))
  # var_cell_ks = ks.test(var_cell_predictions, var_cell_labels)$statistic
  var_cell_ks = chisq.test(var_cell_predictions, var_cell_labels)$statistic

  var_gene_labels = mapply(var, asplit(test_data, 1))
  var_gene_predictions = mapply(var, asplit(predictions, 1))
  # var_gene_ks = ks.test(var_gene_predictions, var_gene_labels)$statistic
  var_gene_ks = chisq.test(var_gene_predictions, var_gene_labels)$statistic

  ks_sum = 0 - (mean_gene_ks + mean_cell_ks + var_gene_ks + var_cell_ks)

  return (list(Score = ks_sum))

}

if (args$study == "zebrafish") {
  train_data = loadCellByGeneData("~/scratch/Structure_VAE_scRNA_Simulator/Data/zebrafish/w_preprocess_new_genes/train_data.mtx")
  test_data = loadCellByGeneData("~/scratch/Structure_VAE_scRNA_Simulator/Data/zebrafish/w_preprocess_new_genes/test_data.mtx")
}
if (args$study == "WOT") {
  train_data = loadData("~/scratch/Structure_VAE_scRNA_Simulator/Data/WOT/w_preprocess/training_all.mtx")
  test_data = loadData("~/scratch/Structure_VAE_scRNA_Simulator/Data/WOT/w_preprocess/testing_all.mtx")
}
if (args$study == "new_mouse_cell") {
  train_data = loadCellByGeneData("~/scratch/Structure_VAE_scRNA_Simulator/Data/new_mouse_cell/w_preprocess_new_genes/train_data.mtx")
  test_data = loadCellByGeneData("~/scratch/Structure_VAE_scRNA_Simulator/Data/new_mouse_cell/w_preprocess_new_genes/test_data.mtx")
}

# TODO: need to smooth data to enable reliable ks test (does not work with discrete data)
train_data = round(train_data) # rounding for model input

trainGeneFilter = which(rowSums(train_data) > 0)
testGeneFilter = which(rowSums(test_data) > 0)
totalGeneFilter = intersect(trainGeneFilter, testGeneFilter)
print(sprintf("Number of genes kept only in training data: %s", length(trainGeneFilter)))
print(sprintf("Number of genes kept only in testing data: %s", length(testGeneFilter)))
print(sprintf("Number of genes kept overall: %s", length(totalGeneFilter)))


print('Before filtering: ')
print(paste("training data dim", c(dim(train_data)[1], dim(train_data)[2])))
print(paste("testing data dim", c(dim(test_data)[1], dim(test_data)[2])))

train_data = train_data[totalGeneFilter,]
test_data = test_data[totalGeneFilter,]

print("remaining zero-count genes in train and test: ")
print(which(rowSums(train_data) == 0))
print(which(rowSums(test_data) == 0))

print("After filtering :")
print(paste("training data dim", dim(train_data)[1], 'x', dim(train_data)[2]))
print(paste("testing data dim", dim(test_data)[1], 'x', dim(test_data)[2]))

n_cell_new = dim(test_data)[2]

library(doParallel)

n_cores = parallel::detectCores()
print(sprintf('number of cores: %s', n_cores))

cl <- makeCluster(n_cores)
registerDoParallel(cl)
clusterExport(cl,c('train_data','test_data', 'modelPrediction', 'n_cell_new'))
clusterEvalQ(cl,expr= {
  library(xgboost)
  library(ParBayesianOptimization)
  library(argparse)
  library(dgof)
  library(ggplot2)
  library(Matrix)
  library(BiocParallel)
  library(zinbwave)
})

bounds <- list(
  latent_size = c(2L, args$max_latent_size)
)

start_time = Sys.time()

print('started tuning')
print('Warnings in tuning:')

optObj <- withCallingHandlers({
  bayesOpt(
  FUN = objective,
  bounds = bounds,
  initPoints = 5,
  iters.n = args$n,
  iters.k = 1,
  parallel=FALSE
  )
}, warning = function(w) {
  print(w)
  invokeRestart('muffleWarning')
})

stopCluster(cl)
registerDoSEQ()


print('finished tuning')

elapsed_time  = as.numeric(Sys.time() - start_time)

best_params = getBestPars(optObj)
optObj$scoreSummary$Score = -optObj$scoreSummary$Score
best_value = min(optObj$scoreSummary$Score)

date_and_time = format(Sys.time(), paste(Sys.Date(), "at %H-%M-%S"))

path = paste("/users/srajakum/scratch/Structure_VAE_scRNA_Simulator/Models/tuning_results/ZINB-WaVE/", args$stu, "_", date_and_time, sep="")
dir.create(path)

print('saving info')

sink(paste(path, "tuning_info.txt", sep="/") , append=TRUE)
cat(paste("Tuning time:", elapsed_time))
cat('\n')
cat(paste("Number of elapsed trials:", args$n))
cat('\n')
cat("Metric type: KS sum")
cat('\n')
cat(paste("Best value:", best_value))
cat('\n')
cat(paste("Best hyperparameters:"))
cat('\n')
for (param in names(best_params)) {
  cat(paste(param, ": ", best_params[[param]], sep = ""))
  cat('\n')
}
sink()

fig = plot(optObj)
ggsave(paste(path, "opt_hist.png", sep="/"))

print('finished saving plot')



