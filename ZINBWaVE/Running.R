"
Description:
  Model fitting and prediction for ZINB-WaVE.
  Codes are adopted from https://github.com/HelenaLC/simulation-comparison/tree/master/code

Author:
  Jiaqi Zhang

Notes:
  If zinbwave package cannot be installed successfully, try this:

    library(withr)
    with_makevars(c(PKG_CFLAGS = '-std=c99'),
              BiocManager::install('zinbwave'),
              assignment = '+=')
"
suppressPackageStartupMessages({
  library(zinbwave)
  library(Matrix)
  library(BiocParallel)
})

modelFitting <- function (data, latent_size = NULL) {
  # Fit ZINB-WaVE model
  print('Started Model Fitting')
  # using latent size value as a flag because I don't know what it's default value is
  if (is.null(latent_size)) {
    model <- zinbFit(data, verbose = TRUE, BPPARAM=MulticoreParam(3)) 
  } else {
    model <- zinbFit(data, K = latent_size, verbose = FALSE, BPPARAM=MulticoreParam(3)) 
  }
  return (model)
}


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