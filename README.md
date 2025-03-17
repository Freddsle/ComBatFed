
# Run FedComBat

## Prerequisite
To run FedComBat, you should install Docker and FeatureCloud pip package:

```bash
pip install featurecloud
```

Then either download FedComBat image from the FeatureCloud docker repository:

```bash
featurecloud app download featurecloud.ai/fedcombat
```
Or build the app locally:

```bash
featurecloud app build featurecloud.ai/fedcombat
```


# Run FedComBat in the test-bed

You can run FedComBat as a standalone app in the FeatureCloud test-bed or FeatureCloud Workflow. You can also run the app using CLI:

```bash
featurecloud controller start --data-dir=./datasets


featurecloud test start --app-image featurecloud.ai/FedComBat --client-dirs './sample/c1,./sample/c2' --generic-dir './sample/generic'

featurecloud test start --app-image FedComBat --client-dirs 'datasets' --generic-dir './sample/generic'
```


## Setting Up the Environment

To recreate the environment, run:

```bash
mamba env create -f environment.yml
```

To activate the environment, run:

```bash
mamba activate FedComBat
```


# Configuration File for FedComBat
The configuration file must be written in YAML and placed in the input folder (default: mnt/input). The file should be named either config.yml or config.yaml unless a custom filename is specified when initializing the client.

The configuration settings must be nested under the top-level key FedComBat.

Example Config File (config.yml):

```yaml
FedComBat:
  data_filename: "data_matrix.tsv"              # Main data file: either features x samples or samples x features.
  design_filename: "design.tsv"           # Optional design matrix: samples x covariates.
                                                    # Must have first column as sample indices.
                                                    # It is read in the following way:
                                                    # pd.read_csv(design_file_path, sep=design_separator, index_col=0)
  data_separator: "\t"                          # Delimiter for the data file.
  design_separator: "\t"                        # Delimiter for the design file.
  min_samples: 5                                # Minimum non-NaN samples required per feature.
  covariates: ["age", "gender", "treatment"]    # List of covariates to be used.
  smpc: true                                    # Secure multi-party computation flag.
  rows_as_features: false                       # Set to true if the data file is an expression file.
  index_col: 0                                  # Column index to use as the data index (0-based).
  position: 1                                   # Client position (if applicable).
  batch_col_name: "batch"                       # Column in design file that contains batch information.
  reference_batch: "Batch_A"                        # Optional. Reference batch for processing, if multiple     
                                                    # batches are present.
  mean_only: false                              # Set to true to perform ComBat without the empirical Bayes step. Default is false.
  parametric: true                              # Set to true to use parametric ComBat. Default is true.
  empirical_bayes: true                         # Set to true to use empirical Bayes step for ComBat. Default is true.
```


# Input Data

The input data must be placed in the input folder. The data file filename must be specified in the configuration file.  
The data file should be either features x samples or samples x features.  
The design file is optional and its filename should be specified in the configuration file. The design file should be with samples x covariates. The first column should contain the sample names, that match the sample names in the data file.

No missing values are allowed in the data file. If missing values are present, the rows with them should be removed before running the app.  
Rows, where all values are zero, will be removed from the data file before running the app.
