import pandas as pd
import numpy as np
import os
import bios
from pathlib import Path
import logging
from typing import Any, Dict, List, Optional, Tuple

class Client:
    def __init__(self) -> None:
        # Initialize logger with module-specific settings
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        # Client configuration attributes
        self.client_name: str = ""
        self.smpc: bool = False
        self.position: Optional[int] = None
        self.rows_as_features: bool = True
        self.rawdata: Optional[pd.DataFrame] = None
        self.data: Optional[pd.DataFrame] = None
        self.data_corrected: Optional[pd.DataFrame] = None
        self.data_separator: Optional[str] = None
        self.design: Optional[pd.DataFrame] = None
        self.design_separator: str = "\t"
        self.variables: Optional[list] = None
        self.batch_labels: list = []
        self.min_samples: int = 0
        self.eb_param: Optional[bool] = True
        self.parametric: Optional[bool] = True
        self.TEST_MODE: Optional[bool] = False

    def read_config(self, config: Dict[str, Any], input_folder: Path, client_name: str) -> Tuple[str, Optional[str], Optional[int]]:
        """
        Reads the config file and sets the client variables.
        
        Returns:
            Tuple containing:
            - data file path (str)
            - design file path (Optional[str])
            - index column (Optional[int])
        """
        # Validate min_samples
        min_samples = config.get("min_samples", 0)
        try:
            self.min_samples = int(min_samples)
        except (ValueError, TypeError) as e:
            raise ValueError("min_samples must be an integer") from e
        self.logger.info("min_samples set to %d", self.min_samples)

        # Read covariates and smpc flag
        self.variables = config.get("covariates")
        self.smpc = config.get("smpc", True)

        # Determine design file path if provided
        design_file_path: Optional[str] = None
        if "design_filename" in config:
            design_file_path = f"{input_folder}/{config['design_filename']}"
        self.design_separator = config.get("design_separator", "\t")

        # Validate expression/data file and its separator
        if "data_filename" not in config:
            raise RuntimeError("No data_filename was given in the config, cannot continue")
        if "data_separator" not in config:
            raise RuntimeError("No separator was given in the config, cannot continue")
        datafile_path = f"{input_folder}/{config['data_filename']}"
        self.data_separator = config["data_separator"]

        # Read expression file flag and index_col
        self.rows_as_features = config.get("rows_as_features", False)
        index_col = config.get("index_col")

        # Validate position
        position = config.get("position")
        if position is not None and not isinstance(position, int):
            raise ValueError("Position must be an integer")
        self.position = position

        # eb_param check
        self.eb_param = config.get("eb_param", True)
        if not isinstance(self.eb_param, bool):
            raise ValueError("eb_param must be a boolean")

        # Validate batch column, if provided
        self.batch_col = config.get("batch_col_name")
        if self.batch_col is not None:
            if not isinstance(self.batch_col, str):
                raise ValueError("Batch column must be a string")
            if self.batch_col and design_file_path is None:
                raise ValueError("Batch column was given but no design file was provided")
            
        # Get combatch parameters
        self.mean_only = config.get("mean_only", False)
        self.parametric = config.get("parametric", True)
        self.empirical_bayes = config.get("empirical_bayes", True)
        
        self.reference_batch = config.get("reference_batch", None)
        if self.reference_batch is not None:
            logging.info("ComBat will ignore the reference batch during correction, only positional information will be used.")
        self.reference_batch = False
        return datafile_path, design_file_path, index_col

    def open_dataset(self, index_col: Optional[int], datafile_path: str, design_file_path: Optional[str]) -> None:
        """
        Opens the dataset and design file, then processes the data into features x samples.
        Only numerical values are considered.
        """
        self.logger.info("Opening dataset %s", datafile_path)
        self.design = None
        self.batch_labels = [self.client_name]

        # Process design file if available
        if design_file_path:
            try:
                design_df = pd.read_csv(design_file_path, sep=self.design_separator, index_col=0)
            except Exception as e:
                raise RuntimeError(f"Error reading design file at {design_file_path}: {e}")
            
            relevant_cols = []
            if self.batch_col:
                relevant_cols.append(self.batch_col)
            if self.variables:
                relevant_cols.extend(self.variables)
            missing_cols = [col for col in relevant_cols if col not in design_df.columns]
            if missing_cols:
                # print head of design_df
                self.logger.error("Design file head:\n%s", design_df.head())
                raise ValueError(f"Design file is missing required columns: {missing_cols}")
            
            self.design = design_df[relevant_cols]
            if self.batch_col:
                unique_batches = self.design[self.batch_col].unique()
                self.batch_labels = [f"{self.client_name}|{batch}" for batch in unique_batches]

        # Read the data file
        csv_kwargs = {"sep": self.data_separator, "index_col": index_col}
        if index_col is None:
            csv_kwargs["index_col"] = 0
        try:
            self.rawdata = pd.read_csv(datafile_path, **csv_kwargs)
        except Exception as e:
            file_type = "expression file" if self.rows_as_features else "CSV file"
            raise RuntimeError(f"Error reading {file_type} at {datafile_path}: {e}")
        self.logger.info("Shape of rawdata: %s", self.rawdata.shape)
        self.data = self.rawdata.copy()

        # Data Cleanup: drop all-NaN rows/columns and enforce numerical types
        self.logger.info("Cleaning up data, removing all-NaN rows and columns, removing all-zero rows")
        self.logger.info("Shape of data before cleanup: %s", self.data.shape)
        self.data = self.data.dropna(axis=1, how='all').dropna(axis=0, how='all')
        # drop all rows where all values are zeros
        self.data = self.data.loc[(self.data != 0).any(axis=1)]
        # remove row where all data are constant (all values are the same)
        self.data = self.data[self.data.nunique(axis=1) > 1]
        self.logger.info("Shape of data after cleanup: %s", self.data.shape)
        self.variables_in_data = False

        # Process data based on file type
        if not self.rows_as_features:
            # Remove variables from data if they are not in the design file
            if self.variables and not design_file_path:
                self.data = self.data.drop(columns=self.variables, errors='ignore')
                self.variables_in_data = True
            if self.data.empty:
                raise ValueError(f"Client {self.client_name}: Error loading data.")
            self.data = self.data.select_dtypes(include=np.number)
            self.num_excluded_numeric = len(self.rawdata.columns) - len(self.data.columns)
            self.data = self.data.T
        else:
            tmp = self.data.T.copy()
            if self.variables and not design_file_path:
                self.variables_in_data = True
                tmp = tmp.drop(columns=self.variables, errors='ignore')
            tmp = tmp.select_dtypes(include=np.number)
            self.num_excluded_numeric = len(self.data.columns) - len(tmp.columns)
            self.data = tmp.T
            # convert to numeric
        
        self.feature_names = list(self.data.index)
        self.sample_names = list(self.data.columns)
        
        if self.design is not None and self.sample_names != list(self.design.index):
            # if any in data index are in design index
            if any([sample in self.design.index for sample in self.data.index]):
                self.logger.error("Check config file for correct 'rows_as_features' setting")
            raise ValueError(
                f"Client {self.client_name}: Sample names in data and design matrix do not match or are not in the same order"
            )
        self.n_samples = len(self.sample_names)
        self.logger.info("Finished loading data, shape: %s, num_features: %d, num_samples: %d",
                         self.data.shape, len(self.feature_names), self.n_samples)

    def config_based_init(self, client_name: str,
                          config_filename: str = "",
                          input_folder: str = os.path.join("mnt", "input")) -> None:
        """
        Initializes a client based on a given config file.
        The config file must be named config.yml or config.yaml unless specified otherwise.
        """
        input_folder_path = Path(input_folder)
        config_files = [config_filename] if config_filename else ["config.yml", "config.yaml"]
        last_exception = None

        for filename in config_files:
            try:
                config = bios.read(f"{input_folder}/{filename}")
                break
            except FileNotFoundError as e:
                last_exception = e
        else:
            raise RuntimeError(f"Could not read the config file. Last error: {last_exception}")
        self.logger.info("Got the following config:\n%s", config)
        
        if "FedComBat" not in config:
            raise RuntimeError("Incorrect format of your config file, the key 'FedComBat' must be in your config file")
        config = config["FedComBat"]
        self.client_name = client_name
        datafile_path, design_file_path, index_col = self.read_config(config, input_folder_path, client_name)
        
        try:
            self.open_dataset(index_col, datafile_path, design_file_path)
        except Exception as e:
            raise ValueError(f"Client {self.client_name}: Error loading dataset. {e}")
        
        assert isinstance(self.data, pd.DataFrame)
        assert self.feature_names is not None
        if self.design is not None:
            assert self.design is not None
            assert self.variables is not None

    def get_batch_feature_presence_info(self, min_samples: int) -> Dict[str, List[str]]:
        """
        Returns information about batches and which features are available on this client
        for each batch. Additionally, it transforms the data by setting features to NaN if they do not meet
        privacy requirements.

        Args:
            min_samples: Minimum number of non-NaN samples required for a feature to be considered for correction.
        
        Returns:
            A dict mapping batch names to a list of hashed feature names that are considered valid for that batch.
        """
        if not min_samples:
            min_samples = self.min_samples
        # Validate required attributes
        if not isinstance(self.data, pd.DataFrame):
            error_msg = "Data must be a pandas DataFrame."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        if not isinstance(self.feature_names, list):
            error_msg = "feature_names must be a list."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        if self.design is None or not isinstance(self.design, pd.DataFrame):
            error_msg = "Design must be a pandas DataFrame."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Log initial data info
        self.logger.info(f"Feature count: {len(self.feature_names)}")

        # Check that feature_names match the DataFrame index (if possible)
        if not set(self.feature_names).issubset(set(self.data.index)):
            missing_features = set(self.feature_names) - set(self.data.index)
            error_msg = f"The following feature names are not in the data index: {missing_features}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Initialize dictionary for batch feature presence info
        batch_feature_presence_info = {batch: [] for batch in self.batch_labels}

        # 1. Privacy check per batch: Ensure that each batch has at least 2 non-NaN samples per feature.
        if self.batch_col:
            # transform values in self.design[self.batch_col] into string to avoid type mismatch
            self.design[self.batch_col] = self.design[self.batch_col].astype(str)
            for batch_label in self.batch_labels:
                # Extract the pure batch name from the label (expected format: "client|batch")
                if "|" in batch_label:
                    parts = batch_label.split("|")
                    if len(parts) < 2:
                        error_msg = f"Batch label '{batch_label}' is not formatted correctly."
                        self.logger.error(error_msg)
                        raise ValueError(error_msg)
                    pure_batch_name = parts[1]
                else:
                    pure_batch_name = batch_label
                
                # Get batch indices from the design matrix
                if self.batch_col not in self.design.columns:
                    error_msg = f"Batch column '{self.batch_col}' not found in design matrix."
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
                batch_indices = self.design[self.design[self.batch_col] == pure_batch_name].index.tolist()
                if not batch_indices:
                    error_msg = f"No indices found in design for batch '{pure_batch_name}'."
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)

                # Check data subset for the batch
                try:
                    batch_data = self.data.loc[:, batch_indices]
                except Exception as e:
                    error_msg = f"Error accessing batch data for indices {batch_indices}: {e}"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
                
                non_nan_counts = batch_data.notnull().sum(axis=1)
                
                # Identify features with exactly 1 non-NaN sample in this batch
                ignore_features = non_nan_counts[non_nan_counts == 1].index.tolist()
                if ignore_features:
                    self.logger.info(
                        f"Ignoring {len(ignore_features)} features for batch {batch_label} due to privacy reasons (exactly one non-NaN sample)."
                    )
                # Set the values for these features in this batch to NaN
                self.data.loc[ignore_features, batch_indices] = np.nan

                # Log the number of features completely dropped after batch-level privacy check
                before_drop_shape = self.data.shape[0]
                self.data = self.data.dropna(axis=0, how='all')
                after_drop_shape = self.data.shape[0]
                self.logger.info(f"Dropped {before_drop_shape - after_drop_shape} features completely empty after processing batch '{batch_label}'.")

        # 2. Global privacy check: Features must have at least min_samples non-NaN entries across the client.
        non_nan_counts = self.data.notnull().sum(axis=1)
        ignore_features_global = non_nan_counts[non_nan_counts < min_samples].index.tolist()
        if ignore_features_global:
            self.logger.info(
                f"Ignoring {len(ignore_features_global)} features for client {self.client_name} due to privacy reasons (fewer than {min_samples} non-NaN samples)."
            )
        self.data = self.data.drop(index=ignore_features_global, errors='ignore')

        # 3. Determine feature presence for each batch.
        for batch_label in self.batch_labels:
            if self.batch_col:
                pure_batch_name = batch_label.split("|")[1] if "|" in batch_label else batch_label
                batch_indices = self.design[self.design[self.batch_col] == pure_batch_name].index.tolist()
                if not batch_indices:
                    error_msg = f"No indices found in design for batch '{pure_batch_name}' during feature presence determination."
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
                try:
                    batch_data = self.data.loc[:, batch_indices]
                except Exception as e:
                    error_msg = f"Error accessing batch data for batch '{batch_label}': {e}"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
            else:
                batch_data = self.data

            self.logger.info(f"Checking feature presence in batch '{batch_label}' with {batch_data.shape[1]} samples.")
            for feature in self.feature_names:
                # If the feature exists and at least one sample is non-NaN in this batch
                if feature in batch_data.index:
                    if not batch_data.loc[feature].isnull().all():
                        batch_feature_presence_info[batch_label].append(feature)
                else:
                    self.logger.warning(f"Feature '{feature}' not found in data index during batch '{batch_label}' processing.")

        # Final check: Raise an error if no features are available for correction
        if not any(batch_feature_presence_info.values()):
            error_msg = "No features available for correction. Check the data and design matrix after privacy filtering."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        return batch_feature_presence_info

    def validate_inputs(self, global_variables: list) -> None:
        """
        Validates the client's expression/covariate inputs by ensuring that the client's variables
        align with the global variables.

        Args:
            global_variables: List of global covariate names.
        
        Raises:
            ValueError: If validation fails in validate_variables.
        """
        self.validate_variables(global_variables)
        self.validate_data()
        self.logger.info(f"Client {self.client_name}: Inputs validated.") 

    def validate_data(self) -> None:
        """
        Validates the client's data matrix and design matrix (if available).

        Raises:
            ValueError: If the data is not a DataFrame, if feature names are not loaded, or if variables are not loaded.
        """
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("self.data must be a pandas DataFrame")
        if not self.feature_names:
            raise Exception("Feature names are not loaded yet, cannot set data")
        
        # if any NA values in the data matrix
        if self.data.isnull().values.any():
            raise ValueError("Data matrix contains NA values. Please clean the data and rerun ComBat.")
        
        # if variables in design have the same values for all samples - error covariate confound with batch
        if self.variables:
            if self.design is None:
                raise ValueError("Design matrix must be loaded to validate variables")
            for var in self.variables:
                if len(self.design[var].unique()) == 1:
                    logging.info("The covariate is confounded with batch! Remove the covariate and rerun ComBat.")
                    raise ValueError(f"Variable {var} has the same value for all samples in the design matrix. Check for confounding with batch.")
                # if two columns of the design matrix are the same - error covariate confound between covariates
                for var2 in self.variables:
                    if var != var2 and self.design[var].equals(self.design[var2]):
                        logging.info("The covariates are confounded! Remove the covariate and rerun ComBat.")
                        raise ValueError(f"Variables {var} and {var2} have the same values for all samples in the design matrix. Check for confounding between covariates.")
        self.logger.info("Client %s: Data validated", self.client_name)
            
    def validate_variables(self, global_variables: list) -> None:
        """
        Validates and updates the client's covariate variables based on the global intersection.
        
        The function ensures that:
        - If no global variables are provided but local ones exist, local variables are ignored.
        - If global variables are provided but the client does not have any local covariates,
            an error is raised.
        - Extra local covariates that are not present globally are ignored with a warning.
        - Finally, it checks that each of the resulting covariates exists either in the design
            matrix (if available) or in the data.
        
        Args:
            global_variables: List of global covariate names.
        
        Raises:
            ValueError: If required covariates are missing either locally or in the design/data.
        """
        # Validate required types.
        if not isinstance(self.variables, list):
            raise ValueError("Variables must be a list.")
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")
        
        # Handle case: No global covariates provided.
        if global_variables is None:
            self.logger.info(
                "No common global covariates selected; Any local covariates will be ignored."
            )
            self.variables = None
            return
        # If global covariates are provided but the client has no local covariates.
        if self.variables is None:
            raise ValueError("Global covariates were selected but this client does not have any local covariates.")
        
        # Identify differences between local and global variables.
        extra_global = set(global_variables) - set(self.variables)
        extra_local = set(self.variables) - set(global_variables)
        if extra_global:
            raise ValueError(f"Global covariates selected that are not available in this client's covariates: {extra_global}")
        if extra_local:
            self.logger.warning(f"Client {self.client_name}: {len(extra_local)} extra local covariates not available globally will be ignored.")
        
        # Update client's variables to only include the global intersection.
        self.variables = global_variables
        
        # Verify that each covariate exists in the design matrix (if available) or in the data.
        if self.design is not None:
            missing_in_design = [var for var in self.variables if var not in self.design.columns]
            if missing_in_design:
                raise ValueError(f"Variables not found in the design matrix: {missing_in_design}")
        else:
            missing_in_data = [var for var in self.variables if var not in self.data.index.tolist()]
            if missing_in_data:
                raise ValueError(f"Variables not found in the data: {missing_in_data}")


    def set_data(self, global_feauture_names: List[str]) -> None:
        """
        Aligns the local data matrix with the global feature set.
        For features found locally but not in the global list, these rows are dropped.
        Finally, the data is reindexed so that the feature order exactly matches global_feauture_names.

        Args:
            global_feauture_names: List of features defining the desired (global) order.
            
        Raises:
            ValueError: If feature names are not loaded or if, after processing, the data index
                does not match the global feature set.
        """
        
        self.logger.info(
            f"Local features: {len(self.feature_names)}; Global features: {len(global_feauture_names)}"
        )
        # Determine features to drop and those to add.
        self.extra_local_features = set(self.feature_names) - set(global_feauture_names)
        self.extra_global_features = set(global_feauture_names) - set(self.feature_names)

        if self.extra_global_features:
            raise ValueError(f"Error in feature alignment: {len(self.extra_global_features)} global features not found locally.")
        
        self.logger.info(
            f"Dropping {len(self.extra_local_features)} extra local features/rows."
        )
        
        # Drop local features not present globally.
        self.data = self.data.drop(index=list(self.extra_local_features), errors='ignore')
        
        # # Add missing global features as rows with NaN.
        # self.data = self.data.reindex(self.data.index.union(list(self.extra_global_features)))
        # self.data.loc[list(self.extra_global_features)] = np.nan
        
        # Validate that the union of features matches the global set.
        if set(global_feauture_names) != set(self.data.index):
            raise ValueError("INTERNAL ERROR: data matrix index does not match the global features list")
        
        # Reorder rows to follow global_feauture_names order.
        self.data = self.data.reindex(global_feauture_names)
        self.feature_names = global_feauture_names


    def create_design(self, cohorts: List[str]) -> Optional[str]:
        """
        Constructs the design matrix for batch effect correction.
        
        The design matrix includes:
        - Batch indicator columns: For each batch, a binary indicator is added.
        - Covariate columns (if provided)
        
        Args:
            cohorts: List of cohort names in the format ["client|batch", ...]. The last entry is treated as the reference batch.
        
        Returns:
            None if successful; otherwise, returns an error message string (e.g., if a privacy error is encountered).
        
        Raises:
            ValueError: If multiple batches exist but no batch column is provided.
        """
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("self.rawdata must be a pandas DataFrame")
        
        # Initialize (or update) the design matrix with the intercept.
        if self.design is None:
            self.design = pd.DataFrame(index=self.data.columns)
        
        # Process cohorts.
        if not cohorts:
            raise ValueError("No cohorts provided")
        
        cohorts_splitlist = [cohort.split("|") for cohort in cohorts]
        
        # Determine if the client has a single batch.
        client_batches = [split[0] for split in cohorts_splitlist if split[0] == self.client_name]
        if len(client_batches) == 1:
            self.logger.info(f"Client {self.client_name} has only one batch")
            for cohort in cohorts:
                self.design[cohort] = 1 if self.client_name == cohort.split("|")[0] else 0
        else:
            # For multiple batches, a valid batch column must be provided.
            if not self.batch_col:
                self.logger.error(
                    f"Client {self.client_name} has multiple batches but no batch column provided. Cohorts: {cohorts}"
                )
                raise ValueError("Batch column was not given but multiple batches for this client were found")
            for idx, cohort_parts in enumerate(cohorts_splitlist):
                cohort_name = cohorts[idx]
                if self.client_name != cohort_parts[0]:
                    self.design[cohort_name] = 0
                else:
                    self.design[cohort_name] = 0
                    sample_indices = self.design[self.design[self.batch_col] == cohort_parts[1]].index
                    self.design.loc[sample_indices, cohort_name] = 1
        
        # Remove the batch column from the design matrix if present.
        if self.batch_col and self.batch_col in self.design.columns:
            self.design = self.design.drop(columns=[self.batch_col])
        
        # Reorder columns: batch columns (based on position in cohorts) followed by covariates.
        desired_order = cohorts

        if self.variables:
            if self.variables_in_data:
                covariate_data = self.rawdata.T[self.variables]
                self.design = self.design.join(covariate_data)
            else:
                if not all(var in self.design.columns for var in self.variables):
                    self.logger.error(
                        f"ERROR: the given variables {self.variables} were not found in the design matrix."
                    )
                    raise ValueError("Variables not found in the design matrix")
            desired_order = list(cohorts) + list(self.variables)

        self.design = self.design.loc[:, desired_order]
        
        # Privacy check: Ensure more samples than design columns.
        if self.design.shape[0] <= self.design.shape[1]:
            if not self.TEST_MODE:
                self.logger.error(
                f"Privacy Error: Insufficient samples for privacy. Samples: {self.design.shape[0]}, Design columns: {self.design.shape[1]}")
                raise ValueError("Privacy Error: Insufficient samples for privacy")
        
        # If any batch contains only one sample - error privacy
        for batch_column in cohorts:
            batch_counts = self.design[batch_column].value_counts()
            if batch_counts.min() == 1:
                self.logger.error("Some batches contain only one sample! Check for privacy.")
                raise ValueError("Some batches contain only one sample. Check for privacy.")
            
        return
    

    #############################
    # ComBat-related functions #
    #############################
    def compute_XtX_XtY(self) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Computes the XtX and XtY matrices for each feature.
        
        For each feature (row in self.data), it selects a sample, then computes:
        - XtX: (X_valid)^T @ (X_valid)
        - XtY: (X_valid)^T @ y_valid

        Returns:
            A tuple (XtX, XtY) where:
            - XtX is an array of shape (n_features, k, k)
            - XtY is an array of shape (n_features, k)
        """
        if not isinstance(self.design, pd.DataFrame):
            self.logger.error("Design matrix must be a pandas DataFrame.")
            raise ValueError("Design matrix must be a pandas DataFrame.")
        if not isinstance(self.data, pd.DataFrame):
            self.logger.error("Data matrix must be a pandas DataFrame.")
            raise ValueError("Data matrix must be a pandas DataFrame.")
        
        X = self.design.values  # shape: (n_samples, k)
        Y = self.data.values    # shape: (n_features, n_samples)
        n_features, _ = Y.shape
        k = X.shape[1]
        
        XtX = np.zeros((n_features, k, k))
        XtY = np.zeros((n_features, k))
        
        for i in range(n_features):
            y = Y[i, :]
            valid = np.isfinite(y)  # boolean array of valid indices
            if valid.any():
                x_valid = X[valid, :]  # corresponding rows in design
                y_valid = y[valid]
                XtX[i] = x_valid.T @ x_valid
                XtY[i] = x_valid.T @ y_valid

        self.XtX = XtX
        self.XtY = XtY
        return XtX, XtY

    def get_sigma_summary(
            self,
            B_hat: np.ndarray,
            ref_size: np.ndarray
        ) -> np.ndarray:
        """
        Computes the sigma summary matrix for the ComBat algorithm.

        Args:
            XtX: The aggregated XtX matrix of shape n x k x k.
            B_hat: The B_hat matrix of shape k x n.
            ref_size: The reference size of the sigma summary matrix of shape n x n.

        Returns:
            The sigma summary matrix of shape n x n.
        """
        # Helper to compute row-wise variance ignoring NaNs with ddof=1 (unbiased estimate)
        # This returns an array of variances (one per row).
        def row_vars(x):
            return np.nanvar(x, axis=1, ddof=1)

        # n_array = np.sum(ref_size)
        n_array = self.data.shape[1] # number of samples
        factor = n_array / (n_array - 1)
        # Explicitly transpose correctly (B_hat shape: k x n_features)
        fitted = (self.design.values @ B_hat).T  # shape: (n_features, n_samples)
        diff = self.data.values - fitted  # shape: (n_features, n_samples)
        var_unbiased = row_vars(diff)  # variance per feature (row)
        var_pooled = var_unbiased / factor

        self.sigma = var_pooled

        return var_pooled  # (n_features,)
    
    def get_standardized_data(
        self, 
        B_hat: np.ndarray, 
        stand_mean: np.ndarray,
        var_pooled: np.ndarray,
        ref_size: np.ndarray
        ):
        """
        Computes the standardized data matrix for the ComBat algorithm.
        
        Args:
            B_hat: The B_hat matrix of shape k x n.
            stand_mean: The mean of the standardized data matrix of shape n_features.
            var_pooled: The pooled variance of the standardized data matrix of shape n_features.
            ref_size: The reference size of the standardized data matrix of shape n_features.
        """
        # get mod_mean
        tmp = self.design.copy()
        tmp.iloc[:, :len(ref_size)] = 0
        # mod.mean: (tmp @ B_hat) is (samples x features); transpose gives (features x samples)
        self.mod_mean = (tmp @ B_hat).T
        self.logger.info("mod_mean shape: %s", self.mod_mean.shape)

        selected_means = []
        positions = []
        for batch in self.batch_labels:
            batch_pos_in_design = self.design.columns.get_loc(batch)
            positions.append(batch_pos_in_design)
            selected_means.extend(
                range(
                    sum(ref_size[:batch_pos_in_design]),
                    sum(ref_size[:batch_pos_in_design + 1])
                )
            )
        selected_means = sorted(selected_means)
        # stand.mean[,1:n.array]
        self.stand_mean = stand_mean[:, selected_means]
        
        # Divide each row by the standard deviation (sqrt of var_pooled); replicating across columns.
        self.logger.info("Standardizing data...")
        self.stand_data = (
            self.data.values - self.stand_mean - self.mod_mean
            ) / np.outer(np.sqrt(var_pooled), np.ones(len(selected_means)))
        self.stand_data.index = self.data.index
        self.logger.info("Data standardized, shape: %s", self.stand_data.shape)


    def get_naive_estimates(self):
        """
        Computes the naive estimates for the ComBat algorithm.
        """
        self.batch_design = self.design.loc[:, self.batch_labels]
        batch_design = self.batch_design.values
        Y = self.stand_data.values    # shape: (n_features, n_samples)        
        gamma_hat = np.matmul(
            np.linalg.inv(
                np.matmul(batch_design.transpose(), batch_design)
            ), batch_design.transpose(),)
        gamma_hat = np.matmul(gamma_hat, Y.transpose())
        self.logger.info("Computed gamma_hat, shape: %s", gamma_hat.shape)

        # Compute delta.hat for each batch in this client.
        # Loop over batches in a fixed order (e.g., sorted by batch label)
        delta_hat_rows = []
        n_features = self.stand_data.shape[0]
        if self.mean_only:
            delta_row = np.ones(n_features)
        else:
            for label in self.batch_labels:
                indices = np.where(self.design[label] == 1)[0]
                # Compute variance for each feature over the columns specified by indices.
                # Use ddof=1 for unbiased estimator; nanvar ignores NaNs.
                delta_row = np.nanvar(self.stand_data.iloc[:, indices], axis=1, ddof=1)
            delta_hat_rows.append(delta_row)
        delta_hat = np.vstack(delta_hat_rows)  # shape: (n_batches, n_features)
        self.logger.info("Computed delta_hat, shape: %s", delta_hat.shape)

        self.gamma_hat = gamma_hat
        self.delta_hat = delta_hat
        self.logger.info("Computed naive estimates.")

    def apriorMat(self, delta_hat: pd.DataFrame) -> pd.Series:
        """
        Computes the apriorMat for the given delta_hat DataFrame.
        
        Parameters:
        -----------
        delta_hat : pd.DataFrame
            A DataFrame with rows as batches and columns as features.
        
        Returns:
        --------
        pd.Series
            A pandas Series where each element corresponds to the computed value 
            (2*s2 + m^2) / s2 for each row, with the index preserved from delta_hat.
        """
        # Compute row-wise means and variances.
        m = delta_hat.mean(axis=1)
        # ddof=1 uses sample variance; adjust if necessary.
        s2 = delta_hat.var(axis=1, ddof=1)
        
        # Compute the apriorMat: (2*s2 + m^2) / s2.
        out = (2 * s2 + m**2) / s2
        return out

    def bpriorMat(self, delta_hat: pd.DataFrame) -> pd.Series:
        """
        Computes the bpriorMat for the given delta_hat DataFrame.
        
        Parameters:
        -----------
        delta_hat : pd.DataFrame
            A DataFrame with rows as batches and columns as features.
        
        Returns:
        --------
        pd.Series
            A pandas Series where each element corresponds to the computed value 
            (m*s2 + m^3) / s2 for each row, with the index preserved from delta_hat.
        """
        # Compute row-wise means and variances.
        m = delta_hat.mean(axis=1)
        s2 = delta_hat.var(axis=1, ddof=1)
        
        # Compute the bpriorMat: (m*s2 + m^3) / s2.
        out = (m * s2 + m**3) / s2
        return out

    def postmean(self, g_hat, g_bar, n, d_star, t2):
        """
        Compute the posterior mean given the parameters.
        
        Parameters:
        - g_hat: Estimated value
        - g_bar: Prior mean value
        - n: Sample size or weight for g_hat
        - d_star: Weight for the prior mean g_bar
        - t2: Scaling factor for the weight of g_hat
        
        Returns:
        - The computed posterior mean.
        """
        return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)

    def biweight_midvar(self, data, center=None, axis=None):
        if center is None:
            center = np.median(data, axis=axis, keepdims=True)
            
        mad = np.median(abs(data - center), axis=axis, keepdims=True)
        
        d = data - center
        u = d/(9*mad)
        
        indic = np.abs(u) < 1
        
        num = d * d * (1. - u**2)**4
        num[~indic] = 0.
        num = np.sum(num, axis=axis)
        
        dem = (1. - u**2) * (1. - 5.*u**2)
        dem[~indic] = 0.
        dem = np.abs(np.sum(dem, axis=axis))**2
        
        n = np.sum(np.ones(data.shape), axis=axis)
        return n * num/dem

    def postvar(self, sum2, n, a, b):
        return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)

    def it_sol(self, sdat, g_hat, d_hat, g_bar, t2, a, b, conv=0.0001, robust=False):
        n = (1 - np.isnan(sdat)).sum(axis=1)
        g_old = g_hat.copy()
        d_old = d_hat.copy()
        ones = np.ones((1, sdat.shape[1]))

        change = 1
        count = 0
        while change > conv:
            g_new = np.array(self.postmean(g_hat, g_bar, n, d_old, t2))
            
            if robust:
                sum2 = n * self.biweight_midvar(
                    sdat, center = g_new.reshape((g_new.shape[0], 1)), axis=1)
                # sum2 = n*(1.482602218505602*np.median(abs(sdat - np.dot(g_new.reshape((g_new.shape[0], 1)), ones)), axis = 1)) ** 2
            else:
                sum2 = ((sdat - np.dot(g_new.reshape((g_new.shape[0], 1)), ones)) ** 2).sum(
                    axis=1
                )
            
            d_new = self.postvar(sum2, n, a, b)
            change = max(max(abs(g_new - g_old) / g_old), max(abs(d_new - d_old) / d_old))            
            g_old = g_new
            d_old = d_new
            count = count + 1
        adjust = (g_new, d_new)
        return adjust

    def int_eprior(self, sdat, g_hat, d_hat):
        import math

        r = sdat.shape[0]
        gamma_star, delta_star = [], []
        for i in range(0, r, 1):
            g = np.delete(g_hat, i)
            d = np.delete(d_hat, i)
            x = sdat[i, :]
            n = x.shape[0]
            j = np.repeat(1, n)
            A = np.repeat(x, g.shape[0])
            A = A.reshape(n, g.shape[0])
            A = np.transpose(A)
            B = np.repeat(g, n)
            B = B.reshape(g.shape[0], n)
            resid2 = np.square(A - B)
            sum2 = resid2.dot(j)
            LH = 1 / (2 * math.pi * d) ** (n / 2) * np.exp(-sum2 / (2 * d))
            LH = np.nan_to_num(LH)
            gamma_star.append(sum(g * LH) / sum(LH))
            delta_star.append(sum(d * LH) / sum(LH))
        adjust = (gamma_star, delta_star)
        return adjust

    def get_parametric_estimators(self, t2, a_prior, b_prior):
        """
        Compute parametric EB estimators based on naive estimators.

        Parameters:
        t2 : Variance of gamma_hat (per feature).
        a_prior : a_priorMat (per feature).
        b_prior : b_priorMat (per feature).
        """

        delta_hat = self.delta_hat
        s_data = self.stand_data.values
        gamma_hat = self.gamma_hat.copy()
        gamma_bar = self.gamma_bar.copy()

        gamma_star_rows = []
        delta_star_rows = []
        for i, label in enumerate(self.batch_labels):
            if self.mean_only:
                gamma_star_row = self.postmean(gamma_hat[i, :], gamma_bar[i], 1, 1, t2[i])
                delta_star_row = np.ones(s_data.shape[0])
            else:
                indices = np.where(self.design[label] == 1)[0]
                temp = self.it_sol(
                    s_data[:, indices],
                    gamma_hat[i, :], 
                    delta_hat[i, :],
                    gamma_bar[i], 
                    t2[i], 
                    a_prior[i], 
                    b_prior[i]
                )
                gamma_star_row = temp[0]
                delta_star_row = temp[1]
            
            gamma_star_rows.append(gamma_star_row)
            delta_star_rows.append(delta_star_row)
        
        self.gamma_star = np.vstack(gamma_star_rows)
        self.delta_star = np.vstack(delta_star_rows)
    
    def get_nonparametric_estimators(self):
        """
        Compute non-parametric EB estimators based on naive estimators.
        """
        delta_hat = self.delta_hat
        s_data = self.stand_data.values
        gamma_hat = self.gamma_hat

        gamma_star_rows = []
        delta_star_rows = []
        for i, label in enumerate(self.batch_labels):
            if self.mean_only:
                delta_hat[i, :] = 1
            indices = np.where(self.design[label] == 1)[0]
            temp = self.int_eprior(s_data.iloc[:, indices],
                               gamma_hat[i, :], delta_hat[i, :])
            gamma_star_rows.append(temp[0])
            delta_star_rows.append(temp[1])
        self.gamma_star = np.vstack(gamma_star_rows)
        self.delta_star = np.vstack(delta_star_rows)

    def get_eb_estimators(self):
        """
        Compute empirical Bayes (EB) estimators based on naive estimators.
        
        Parameters:
        naive_estimators : Dictionary with "gamma.hat" and "delta.hat" (arrays: n_batches x n_features)
        s_data           : Standardized data (features x samples)
        data_dict        : Dictionary with keys "batches", "n.batch", "ref.batch", "ref"
        parametric       : Boolean; if True use parametric EB, otherwise non-parametric.
        mean_only        : Boolean.
        
        Returns:
        A dictionary with EB estimators: "gamma.star", "delta.star", "gamma.bar", "t2", "a.prior", "b.prior"
        """        
        # Compute gamma.bar as the mean across batches (axis 0 of gamma_hat, per feature)
        self.gamma_bar = np.nanmean(self.gamma_hat, axis=1)
        # Compute t2 as the variance of gamma_hat (per feature)
        # t2 = np.nanvar(self.gamma_hat, axis=0, ddof=1)
        t2 = self.gamma_hat.var(ddof=1, axis=1)
        self.t2 = t2
        
        a_prior = self.apriorMat(self.delta_hat)
        b_prior = self.bpriorMat(self.delta_hat)

        self.a_prior = a_prior
        self.b_prior = b_prior
        self.logger.info("Computed a_prior and b_prior.")

        if self.parametric:
            self.get_parametric_estimators(t2, a_prior, b_prior)
        else:
            self.get_nonparametric_estimators()

        self.logger.info("Computed EB estimators.")


    def get_corrected_data(self, var_pooled):
        """
        Compute the corrected (combat-adjusted) data.
        
        Parameters:
        var_pooled      : Pooled variance of the standardized data matrix (features x samples).

        Returns:
        bayesdata       : The corrected data matrix.
        """
        mod_mean = self.mod_mean
        mod_mean.index = self.data.index
        stand_mean = self.stand_mean
        batch_design = self.batch_design        
        if self.eb_param:
            gamma_star = self.gamma_star
            delta_star = self.delta_star
        else:
            gamma_star = self.gamma_hat
            delta_star = self.delta_hat

        bayesdata = self.stand_data.copy()
        j = 0
        
        self.logger.info("Correcting data...")
        for label in self.batch_labels:
            indices = np.where(self.design[label] == 1)[0]
            fitted = np.dot(batch_design.iloc[indices], gamma_star).T 
            top = bayesdata.iloc[:, indices] - fitted
            bottom = np.outer(np.sqrt(delta_star[j, :]), np.ones(len(indices)))
            bayesdata.iloc[:, indices] = top / bottom
            j += 1 
        self.top = top
        self.bottom = bottom
        # Re-scale and add back the mean adjustments.
        self.logger.info("Rescaling and adding back mean adjustments...")
        self.logger.info("Data shape: %s", bayesdata.shape)
        self.logger.info("var_pooled shape: %s", var_pooled.shape)
        self.logger.info("stand_mean shape: %s", stand_mean.shape)
        self.logger.info("mod_mean shape: %s", mod_mean.shape)
        self.logger.info("indices shape: %s", len(indices))

        self.pooled_var = var_pooled
        
        self.tmp_bayesdata = bayesdata.copy()
        # Rescale the data using the pooled variance and add back the mean adjustments.
        bayesdata = (
            bayesdata * np.outer(np.sqrt(var_pooled), np.ones(len(indices)))
        ) + stand_mean + mod_mean
        self.logger.info("Rescaling completed.")
        bayesdata.index = self.data.index
        self.logger.info("Data corrected, shape: %s", bayesdata.shape)
        
        return bayesdata
    

    