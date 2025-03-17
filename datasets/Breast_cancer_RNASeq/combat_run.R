library(sva)
library(tidyverse)

data_path <- "datasets/Breast_cancer_RNASeq/"
datasets <- c("GSE129508", "GSE149276", "GSE58135")

all_expression <- NULL
all_metadata <- NULL

for(dataset in datasets){
  expr_data <- read.table(paste0(data_path, "before/", dataset, "/expr_for_correction.tsv"), 
                          header = TRUE, sep = "\t")
  metadata <- read.table(paste0(data_path, "before/", dataset, "/design.tsv"), 
                         header = TRUE, sep = "\t")
  print(paste("Samples in dataset", dataset, "-", ncol(expr_data)-1))
  # save data
  if(is.null(all_metadata)){
    all_metadata <- metadata
    all_expression <- expr_data
  } else {        
    all_metadata <- rbind(all_metadata, metadata)
    all_expression <- full_join(all_expression, expr_data, by = "gene_id")
  }
  print(paste0("Combined Samples: ", nrow(all_metadata), "; Features: ", nrow(all_expression)))
  print(" ")
}

all_expression <- all_expression %>% column_to_rownames("gene_id")
all_expression <- all_expression[, all_metadata$sample_id]
print(paste0("Number of samples: ", nrow(all_metadata)))
print(paste0("Number of features: ", nrow(all_expression), "; Number of samples: ", ncol(all_expression)))


design <- model.matrix(~all_metadata$lum)

ComBat(dat = all_expression, batch = all_metadata$batch, mod = design)

corrected_expr <- as.data.frame(corrected_expr)
