### Input: Gene expression data structure with .rds extension
### Output: Gene expression matrix of normalized counts per cell 
###   multiplied by scale factor of 10000 with .mtx extension
###   that can be read into Python


library(monocle)
library(Seurat)
library(BiocManager)
library(Matrix)
library(data.table)
library(RGF)
library(stringr)

#if (!requireNamespace("BiocManager", quietly = TRUE))
#install.packages("BiocManager")
#BiocManager::install(version = "3.12")
#BiocManager::install("monocle") ## v2


### Function definition (to be used later)
extract_attributes <- function(gtf_attributes, att_of_interest){
  att <- unlist(strsplit(gtf_attributes, " "))
  if(att_of_interest %in% att){
    return(gsub("\"|;","", att[which(att %in% att_of_interest)+1]))
  }else{
    return(NA)}
}
#
#
####Set directory and parameters
#if (getwd() != 'D:/programming/Python/SC_VAE/R_script'){
#  setwd('D:/programming/Python/SC_VAE/R_script')
#}
##base_path <- getwd()
##setwd(base_path)

scale_factor <- 10000
subsample <- 100000

scale_factor_str <- format(scale_factor, digits=1, scientific=TRUE)
scale_factor_str <- str_replace(scale_factor_str, "[+]", "")


### Read in downloaded data files

cell_metadata <- read.csv('./Data/mouse_cell/cell_annotate.csv')
gene_annotation <- read.csv('./Data/mouse_cell/gene_annotate.csv')

print('done with cell and gene metadata')

cds <- readRDS('./Data/mouse_cell/gene_count_cleaned_sampled_100k.RDS')
sample_size <- subsample
sample_size_str <- paste(toString(as.integer(sample_size/1000)), "k", sep="")

print('done with reading cds')


### Create new Seurat object with time point metadata added

sObj_metadata <- data.frame("cluster"= cell_metadata$Main_Cluster, row.names=cell_metadata$sample)
sObj_metadata["time_pt"] <- cell_metadata$development_stage
sObj_raw <- CreateSeuratObject(cds, meta.data=sObj_metadata)
if (sample_size < 100000) {
  sObj_raw <- subset(sObj_raw, downsample = sample_size)
}

print('done with adding metadata')


### Read in mouse gene annotations and eliminate X, Y chromosome genes

mouse_gtf <- fread("./Data/mouse_cell/gencode.vM26.annotation.gtf")

setnames(mouse_gtf, names(mouse_gtf), c("chr", "source", "feature", "start", "end", "misc1",
                   "mics2", "misc3", "attributes"))
mouse_gtf <- mouse_gtf[feature == "gene"]
mouse_gtf_somatic <- mouse_gtf[chr != "chrX" & chr != "chrY"]
mouse_gtf_x_y <- mouse_gtf[chr == "chrX" | chr == "chrY"]

ensembl_ids_x_y <- unlist(lapply(mouse_gtf_x_y$attributes, extract_attributes, "gene_id"))
ensembl_ids <- unlist(lapply(mouse_gtf_somatic$attributes, extract_attributes, "gene_id"))
gene_names <- lapply(mouse_gtf_somatic$attributes, extract_attributes, "gene_name")
counts_filtered <- GetAssayData(sObj_raw, assay = "RNA")
counts_filtered <- counts_filtered[-(which(rownames(counts_filtered) %in% ensembl_ids_x_y)),]

sObj <- subset(sObj_raw, features = rownames(counts_filtered))


### Normalize Seurat object using RC with scale factor (10000 default)
all.genes <- rownames(sObj)
#sObj <- NormalizeData(sObj, normalization.method='RC', scale.factor=scale_factor) #TODO: normalization

### Select all time points except for 13.5 (last time point)
###   and save into new Seurat object
#sObj_cells_at_tp_not_5 <- subset(x = sObj, subset = time_pt != 13.5) # TODO: all data


### Sanity check
# cluster_check <- FALSE

# if (cluster_check == TRUE) {
#   sObj_cluster_check <- FindVariableFeatures(sObj_all_genes, selection.method = "vst", nfeatures = 2000)
#   
#   # sObj_cluster_check <- ScaleData(sObj_cluster_check, features=all.genes)
#   # sObj_cluster_check <- FindNeighbors(sObj_cluster_check, dims = 1:6)
#   # sObj_cluster_check <- FindClusters(sObj_cluster_check, resolution = 0.5)
#   
#   sObj_cluster_check <- RunPCA(sObj_cluster_check, features = VariableFeatures(object = sObj_cluster_check))
#   DimPlot(sObj_cluster_check, reduction = "pca")
#   DimPlot(sObj_cluster_check, reduction = "pca", group.by="cluster")
#   ElbowPlot(sObj_cluster_check)
#   
#   # sObj_tsne <- RunTSNE(sObj_cluster_check, dims = 1:6)
#   # DimPlot(sObj_tsne, reduction = "tsne")
#   # DimPlot(sObj_tsne, reduction = "tsne", group.by="cluster")
#   
#   sObj_umap <- RunUMAP(sObj_cluster_check, dims = 1:6)
#   DimPlot(sObj_umap, reduction = "umap")
#   DimPlot(sObj_umap, reduction = "umap", group.by="cluster")
#   
#   jpeg('rplot.jpg')
#   DimPlot(sObj_umap, reduction = "umap", group.by="cluster")
#   dev.off()
# }


### Determine top 2000 most highly variable genes from all time points 
###   except last
#sObj_var_genes <- FindVariableFeatures(sObj_cells_at_tp_not_5, selection.method = "vst", nfeatures = 2000) # TODO: data excluding the last time point
sObj_var_genes <- FindVariableFeatures(sObj, selection.method = "vst", nfeatures = 2000)
var_genes <- VariableFeatures(sObj_var_genes)
sObj_var_genes <- sObj[var_genes,]


### Write each time point to .mtx file # TODO: write data

samp_time = c(9.5,10.5,11.5,12.5,13.5)
samp_time_str = c('9_5', '10_5', '11_5', '12_5', '13_5')

for (i in seq(length(samp_time))){
  t <- samp_time[i]
  t_str <- samp_time_str[i]
  cells_at_tp <- subset(x = sObj_var_genes,
                        subset = time_pt == t)
  counts_norm_var <- GetAssayData(cells_at_tp, assay = 'RNA')

  print(length(rownames(counts_norm_var)))
  print(length(colnames(counts_norm_var)))

  #matrix_fname <- paste('gene_exp_mat_time_', t_str, '_',
  #                      sample_size_str, '_sf_', scale_factor_str, '_rc.mtx', sep='')
   matrix_fname <- paste('gene_cnt_mat_time_', t_str, '_',
                        sample_size_str, '_sf_', scale_factor_str, '_rc.mtx', sep='')
  writeMM(counts_norm_var, matrix_fname)

  #cell_fname <- paste('cells_time_', t_str,  '_',
  #                    sample_size_str, '_sf_', scale_factor_str, '_rc.csv', sep='')
  cell_fname <- paste('cells_cnt_time_', t_str,  '_',
                      sample_size_str, '_sf_', scale_factor_str, '_rc.csv', sep='')
  write.csv(list(colnames(counts_norm_var)), cell_fname,
            row.names = FALSE, quote = FALSE)
}