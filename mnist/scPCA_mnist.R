library(here)
library(tidyverse)
library(devtools)
library(scPCA)
# load scPCA implementation
#source(here::here("analyses/dengue_data/cluster_files/scpca.R"))
library(elasticnet)
library(Rtsne)
library(umap)
library(ggpubr)
library(naniar)
library(SingleCellExperiment)
library(cluster)
library(xtable)

script_dir <- dirname(sys.frame(1)$ofile)
data_dir <- file.path(script_dir, "data")

target_df <- read_csv(file.path(data_dir, "MNIST_foreground.csv"))
background_df <- read_csv(file.path(data_dir, "MNIST_background.csv"))

start <- Sys.time()

scpca_mnist <- scPCA(target = target_df,
                    background = background_df,
                    n_centers = 2,
                    scale = TRUE,
                    max_iter = 10)

end <- Sys.time()

runtime <- end - start
print(runtime)

write.csv(scpca_mnist$x, file = "scpca_mnist.csv")
