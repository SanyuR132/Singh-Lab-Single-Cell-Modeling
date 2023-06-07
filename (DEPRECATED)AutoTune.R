"
Description:
  Automatically tune model parameters.

Author:
  Jiaqi Zhang
"

library("xgboost")
library("ParBayesianOptimization")

source("./Models/R_Utils.R")
source("./Models/ZINBWaVE/Running.R")


# https://htmlpreview.github.io/?https://github.com/JSB-UCLA/scDesign2/blob/master/vignettes/scDesign2.html
tunescDesign2 <- function() {
  source("./Models/scDesign2/Running.R")

  validateLoss <- function() {
    # Fitting scDesign2
    model <- modelFitting(train_data, sim_method = 'copula')
    # Make predictions
    print("Model predicting...")
    predictions <- modelPrediction(model, n_cell_new, sim_method = 'copula')
    return (predictions)
  }

  # Load mouse cell data
  filename <- "./Data/mouse_cell/wo_preprocess/training_all.mtx"
  train_data <- loadData(filename)
  filename <- "./Data/mouse_cell/wo_preprocess/validate_all.mtx"
  validate_data <- loadData(filename)
  n_cell_new <- dim(validate_data)[2]

  # Note: Subsampled data due to the low-efficiency of scDesign2 model
  # You may need to use all the data for model training.
  train_data <- train_data[c(1:10), c(1:50)]
  validate_data <- validate_data[c(1:10), c(1:50)]

  print(sprintf("Train data shape : %s", dim(train_data)))
  print(sprintf("Validate data shape : %s", dim(validate_data)))
  pred <- validateLoss()
  print()
}

if (TRUE) {

  tunescDesign2()
  model_args <- list(lambda1 = c(0.00, 1), lambda2 = c(0.00, 1))
  optObj <- bayesOpt(
  FUN = tunescDesign2
  , bounds = model_args
  , initPoints = 6
  , iters.n = 50
  , iters.k = 1
)
}