"
Description:
  Utility functions for R scripts.

Author:
  Jiaqi Zhang
"

loadData <- function(filename){
  # Read data
  #print(paste(replicate(50, "-"), collapse = ""))
  print(sprintf("Filename : %s", filename))
  mtx_data <- readMM(filename)
  mtx_data <- as.matrix(mtx_data) # genes X cells
  shape <- dim(mtx_data)
  n_cells <- shape[2]
  n_genes <- shape[1]
  print(sprintf("Num of cells : %s", n_cells))
  print(sprintf("Num of genes : %s", n_genes))
  return (mtx_data)
}

loadCellByGeneData <- function(filename){
  # Read data
  #print(paste(replicate(50, "-"), collapse = ""))
  print(sprintf("Filename : %s", filename))
  mtx_data <- readMM(filename)
  mtx_data <- as.matrix(mtx_data) # cells X genes
  mtx_data <- t(mtx_data) # genes X cells
  shape <- dim(mtx_data)
  n_cells <- shape[2]
  n_genes <- shape[1]
  print(sprintf("Num of cells : %s", n_cells))
  print(sprintf("Num of genes : %s", n_genes))
  return (mtx_data)
}

filterGene <- function (data, min_cn = 5){
  # Filter out low-expressing genes
  filter <- rowSums(data > min_cn) > min_cn
  table(filter)
  return (list("filter"=filter, "data"=data[filter,])) # data should be (genes, cells)
}


getTimeStr <- function (){
  # Formate the currnt time into a string like 20211222_182513 (year month day _ hour min sec)
  return (format(Sys.time(), "%Y%m%d_%H%M%S"))
}