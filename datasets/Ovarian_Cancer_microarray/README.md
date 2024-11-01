# Microarray dataset

6 datasets : "GSE6008", "GSE26712", "GSE40595", "GSE69428", "GSE36668", "GSE14407".

Preprocessing:
- raw data .CEL files from https://www.ncbi.nlm.nih.gov/geo/ 
- data were normalized and log-transformed using rma function.
- rows were collapsed to gene symbols using WGCNA::collapseRows(maxRowVariance).

Do not contain NA values.

Central run:  
- ComBat with Status (normal / HGSC) as covariates and dataset as batches.
- without NA.

# Structure

In /before folder there are two versions on the dataset.  
- one file for all - all datasets in one tsv (all_metadata, all_expression).
- the save but folder per datasets - for App. Contains additionally design file - covariates info in sutable for App format.

For App log2 transformation must be OFF.

"Expression for correction" file is filtered (keeping only intersected rows) - use this for evaluation.
