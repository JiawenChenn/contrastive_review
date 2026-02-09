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

mice_df <- read_csv(file.path(data_dir, "Data_Cortex_Nuclear.csv"))

# split into target and background
target_df <- mice_df %>%
  dplyr::filter(Treatment == "Saline",
                Behavior == "S/C",
               Genotype == "Control" | Genotype == "Ts65Dn") %>%
  select(-Genotype, -Behavior, -Treatment, -MouseID) %>%
  impute_median_if(is.numeric)
background_df <- mice_df %>%
  dplyr::filter(Treatment == "Saline",
                Behavior == "C/S",
                Genotype == "Control") %>%
  select(-Genotype, -Behavior, -Treatment, -MouseID, -class) %>%
  impute_median_all


# run scPCA for using 40 logarithmically seperated contrastive parameter values
# and possible 20 L1 penalty terms
start <- Sys.time()

scpca_mice <- scPCA(target = target_df[, 1:77],
                    background = background_df,
                    n_centers = 2,
                    scale = TRUE,
                    max_iter = 1000)

end <- Sys.time()

runtime <- end - start
print(runtime)
