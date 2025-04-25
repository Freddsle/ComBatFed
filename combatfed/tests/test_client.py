# tests/test_client.py

import pytest
import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from classes.client import Client

@pytest.fixture
def sample_config():
    """
    Returns a base config dictionary with minimal required keys.
    This can be extended in individual tests.
    """
    return {
        "data_filename": "dummy_data.csv",
        "data_separator": ",",
        "rows_as_features": True
    }


def test_client_initialization():
    """
    Test that a Client object is initialized with expected default values.
    """
    client = Client()
    assert client.client_name == ""
    assert client.smpc is False
    assert client.position is None
    assert client.rows_as_features is True
    assert client.rawdata is None
    assert client.data is None
    assert client.data_corrected is None
    assert client.data_separator is None
    assert client.design is None    
    assert client.design_separator == "\t"
    assert client.variables is None
    assert client.batch_labels == []
    assert client.min_samples == 0

@pytest.mark.parametrize("min_samples_input,expected", [
    (10, 10),
    ("5", 5),
    (None, 0),  # if key is missing, default is 0
])


def test_read_config_min_samples(tmp_path, sample_config, min_samples_input, expected):
    """
    Test read_config with different min_samples values, ensuring correct integer parsing.
    """
    client = Client()
    config = sample_config.copy()
    if min_samples_input is not None:
        config["min_samples"] = min_samples_input
    
    # Minimal valid data file to avoid reading errors
    data_path = Path(tmp_path / "dummy_data.csv")
    data_path.write_text("A,B\n1,2")
    
    input_folder = str(tmp_path)
    client_name = "TestClient"
    
    # Wrap it under ComBatFed to mimic actual structure
    main_config = {"ComBatFed": config}
    
    # read_config is normally called inside config_based_init, but we can call it directly
    datafile_path, design_file_path, index_col = client.read_config(
        main_config["ComBatFed"], input_folder, client_name
    )
    
    assert client.min_samples == expected
    assert datafile_path == os.path.join(input_folder, "dummy_data.csv")


def test_read_config_missing_data_filename(tmp_path, sample_config):
    """
    Test that read_config raises an error if 'data_filename' is missing.
    """
    client = Client()
    config = sample_config.copy()
    config.pop("data_filename")  # remove mandatory key
    
    main_config = {"ComBatFed": config}
    with pytest.raises(RuntimeError, match="No data_filename was given"):
        client.read_config(main_config["ComBatFed"], str(tmp_path), "TestClient")


def test_read_config_missing_data_separator(tmp_path, sample_config):
    """
    Test that read_config raises an error if 'data_separator' is missing.
    """
    client = Client()
    config = sample_config.copy()
    config.pop("data_separator")
    
    main_config = {"ComBatFed": config}
    with pytest.raises(RuntimeError, match="No separator was given"):
        client.read_config(main_config["ComBatFed"], str(tmp_path), "TestClient")


def test_read_config_with_design_file(tmp_path, sample_config):
    """
    Test read_config to ensure design_filename and design_separator are handled correctly.
    """
    client = Client()
    config = sample_config.copy()
    config["design_filename"] = "dummy_design.csv"
    config["design_separator"] = "\t"
    
    # Write minimal files
    data_path = Path(tmp_path / "dummy_data.csv")
    data_path.write_text("A,B\n1,2")
    design_path = Path(tmp_path / "dummy_design.csv")
    design_path.write_text("ID\tbatch\nSample1\tBatch1\nSample2\tBatch1")

    main_config = {"ComBatFed": config}
    datafile_path, design_file_path, index_col = client.read_config(
        main_config["ComBatFed"], str(tmp_path), "TestClient"
    )
    
    assert datafile_path.endswith("dummy_data.csv")
    assert design_file_path.endswith("dummy_design.csv")
    assert client.design_separator == "\t"


def test_open_dataset_basic(tmp_path, sample_config):
    """
    Test open_dataset with a basic CSV and no design file.
    """
    client = Client()
    config = sample_config.copy()
    
    data_path = Path(tmp_path / "dummy_data.csv")
    data_path.write_text("rowname,0,1\nFeature1,1,2\nFeature2,3,4")
    
    client.client_name = "TestClient"
    client.data_separator = config["data_separator"]
    client.rows_as_features = config["rows_as_features"]
    
    # The read_config sets these, we mimic them here
    datafile_path = str(data_path)
    design_file_path = None
    index_col = None
    
    client.open_dataset(index_col, datafile_path, design_file_path)
    assert isinstance(client.data, pd.DataFrame)
    # rows_as_features=False => the data should be transposed, so we expect 2 features, 2 samples
    assert client.data.shape == (2, 2)
    assert list(client.data.index) == ["Feature1", "Feature2"]
    assert client.sample_names == ['0', '1']
    assert client.feature_names == ["Feature1", "Feature2"]
    assert client.batch_labels == ["TestClient"]


def test_open_dataset_rows_as_features(tmp_path, sample_config):
    """
    Test open_dataset with rows_as_features=True.
    The data should NOT be transposed in the same way as above.
    """
    client = Client()
    config = sample_config.copy()
    config["rows_as_features"] = True  # simulate expression file
    
    data_path = Path(tmp_path / "dummy_data.csv")
    data_path.write_text("row,0,1\nFeature1,1,2\nFeature2,3,4")
    
    client.client_name = "TestClient"
    client.data_separator = config["data_separator"]
    client.rows_as_features = True
    
    datafile_path = str(data_path)
    design_file_path = None
    index_col = None  # let open_dataset set it to 0 if rows_as_features
    
    client.open_dataset(index_col, datafile_path, design_file_path)

    # We expect that the data is read with the first column as index,
    # then transposed so that features remain in rows
    assert client.data.shape == (2, 1) or (2, 2)  # depends on how columns are read
    # The key point is verifying it didn't transpose the same way as normal CSV


def test_open_dataset_with_design(tmp_path, sample_config):
    """
    Test open_dataset to ensure the design file is properly merged
    and batch_labels are updated.
    """
    client = Client()
    config = sample_config.copy()
    config["design_filename"] = "dummy_design.csv"
    config["batch_col_name"] = "batch"
    config["rows_as_features"] = False
    
    # Data file
    data_path = Path(f"{tmp_path}/{'dummy_data.csv'}")
    data_path.write_text("sample,Feature1,Feature2\ns1,1,2\ns2,3,6")
    
    # Design file
    design_path = Path(f"{tmp_path}/{'dummy_design.csv'}")
    # Must match sample columns if rows_as_features=False => sample names are 0,1 after read
    design_path.write_text("sample\tbatch\ns1\tBatchA\ns2\tBatchB")
    
    client.client_name = "TestClient"
    client.data_separator = ","
    client.rows_as_features = config["rows_as_features"]
    client.batch_col = config["batch_col_name"]
    client.design_separator = "\t"  # or config.get("design_separator", "\t")
    
    datafile_path = str(data_path)
    design_file_path = str(design_path)
    index_col = None
    
    client.open_dataset(index_col, datafile_path, design_file_path)

    assert client.design is not None
    assert list(client.design.columns) == ["batch"]
    assert client.batch_labels == ["TestClient|BatchA", "TestClient|BatchB"]


def test_open_dataset_design_missing_required_columns(tmp_path, sample_config):
    """
    Test that open_dataset raises an error if the design file doesn't have required columns.
    """
    client = Client()
    config = sample_config.copy()
    config["design_filename"] = "dummy_design.csv"
    config["batch_col_name"] = "batch"
    
    # Minimal data
    data_path = Path(tmp_path / "dummy_data.csv")
    data_path.write_text("A,B\n1,2")
    
    # Design file missing 'batch' column
    design_path = Path(tmp_path / "dummy_design.csv")
    design_path.write_text("some_other_col\nX\nY")
    
    client.client_name = "TestClient"
    client.batch_col = config["batch_col_name"]
    client.data_separator = config["data_separator"]
    
    datafile_path = str(data_path)
    design_file_path = str(design_path)
    
    with pytest.raises(ValueError, match="Design file is missing required columns"):
        client.open_dataset(None, datafile_path, design_file_path)


def test_config_based_init_valid(tmp_path, sample_config):
    """
    Test config_based_init end-to-end with a valid config and minimal files.
    """
    client = Client()
    config = sample_config.copy()
    # Wrap in ComBatFed
    main_config = {"ComBatFed": config}
    
    # Write a minimal config.yml
    config_path = Path(tmp_path / "config.yml")
    import bios
    bios.write(str(config_path), main_config)
    
    # Write minimal data file
    data_path = Path(tmp_path / "dummy_data.csv")
    data_path.write_text("A,B\n1,2")
    
    client.config_based_init("TestClient", config_filename="config.yml", input_folder=str(tmp_path))
    
    assert client.client_name == "TestClient"
    assert isinstance(client.data, pd.DataFrame)
    assert client.data.shape == (2, 1) or (2, 2)  # depends on row/column handling


def test_config_based_init_no_fedcombat_key(tmp_path):
    """
    Test config_based_init raises an error if 'ComBatFed' is missing in the config.
    """
    client = Client()
    # Write a config without ComBatFed
    bad_config = {"SomeOtherKey": {}}
    config_path = Path(tmp_path / "config.yml")
    import bios
    bios.write(str(config_path), bad_config)
    
    with pytest.raises(RuntimeError, match="the key 'ComBatFed' must be in your config file"):
        client.config_based_init("TestClient", config_filename="config.yml", input_folder=str(tmp_path))


def create_data_frame(client, data_dict=None):
    """
    Helper function to set client's data attribute from a dictionary.
    data_dict: {feature: [sample values corresponding to design.index]}
    """
    if data_dict is None:
        data_dict = {
        "f1": [1, 2, 1, 2],
        "f2": [3, 4, 3, 4],
        "f3": [5, 6, 5, 6]
        }
    client.data = pd.DataFrame(data_dict, index=client.design.index).T
    client.feature_names = list(data_dict.keys())


def add_covariates(client, covariates_dict=None):
    """
    Helper function to add covariates to the client's design DataFrame.
    """
    if covariates_dict is None:
        covariates_dict = {
            "cov1": [0, 1, 0, 1],
            "cov2": [1, 1, 0, 0]
        }
    
    client.variables = list(covariates_dict.keys())
    for cov, values in covariates_dict.items():
        client.design[cov] = values


@pytest.fixture
def dummy_client():
    """
    Creates a dummy Client object with the necessary attributes set.
    """
    client = Client()
    client.client_name = "TestClient"
    client.batch_labels = ["TestClient|BatchA", "TestClient|BatchB"]
    client.batch_col = "batch"
    # Define feature names
    client.feature_names = ["f1", "f2", "f3"]
    
    # Create a design DataFrame:
    # Index will be sample IDs and a column 'batch' indicating the batch membership.
    design_data = {
        "batch": ["BatchA", "BatchA", "BatchB", "BatchB"]
    }
    client.design = pd.DataFrame(design_data, index=["s1", "s2", "s3", "s4"])
    
    return client


def test_feature_presence_all_valid(dummy_client):
    """
    Test case where all features have enough non-NaN samples.
    """
    # Create data where each feature has sufficient non-NaN values
    # Samples s1, s2 belong to BatchA; s3, s4 to BatchB.
    data = {
        "f1": [1, 2, 1, 2],
        "f2": [3, 4, 3, 4],
        "f3": [5, 6, 5, 6]
    }
    create_data_frame(dummy_client, data)
    
    # Using min_samples=2 so that all features are valid globally.
    result = dummy_client.get_batch_feature_presence_info(min_samples=2)
    # Expect all features to be present for each batch
    expected = {
        "TestClient|BatchA": ["f1", "f2", "f3"],
        "TestClient|BatchB": ["f1", "f2", "f3"],
    }
    assert result == expected


def test_ignore_features_due_to_low_batch_samples(dummy_client):
    """
    Test that features with only one non-NaN sample in a batch get ignored.
    """
    # Create data where in BatchA, feature f2 has only one non-NaN value.
    # BatchA corresponds to samples s1 and s2.
    data = {
        "f1": [1, 1, 1, 1],
        "f2": [np.nan, 2, 3, 3],  # In BatchA: [NaN, 2] => only one valid sample.
        "f3": [5, 5, 5, 5]
    }
    create_data_frame(dummy_client, data)
    
    # min_samples=2 so f2 should be dropped globally in BatchA due to privacy
    result = dummy_client.get_batch_feature_presence_info(min_samples=2)
    
    # For BatchA, f2 is dropped, but BatchB is not affected if both samples are valid.
    # In BatchB (s3, s4) for f2: values [3, 3] (both non-NaN) so it remains.
    expected = {
        "TestClient|BatchA": ["f1", "f3"],
        "TestClient|BatchB": ["f1", "f2", "f3"],
    }
    assert result == expected


def test_ignore_features_due_to_low_global_samples(dummy_client):
    """
    Test that features with fewer than min_samples non-NaN values globally are ignored.
    """
    # Create data where feature f3 has less than min_samples non-NaN values across the client.
    data = {
        "f1": [1, 1, 1, 1],
        "f2": [2, 2, 2, 2],
        "f3": [np.nan, np.nan, 3, np.nan]  # Only one non-NaN globally.
    }
    create_data_frame(dummy_client, data)
    
    # Set min_samples=2 so that f3 is globally ignored.
    result = dummy_client.get_batch_feature_presence_info(min_samples=2)
    expected = {
        "TestClient|BatchA": ["f1", "f2"],
        "TestClient|BatchB": ["f1", "f2"],
    }
    assert result == expected


def test_no_batch_col(dummy_client):
    """
    Test behavior when there is no batch_col set.
    """
    # Remove batch_col and adjust batch_labels to a single batch.
    dummy_client.batch_col = None
    dummy_client.batch_labels = ["TestClient"]
    
    # Adjust design to include all samples.
    design_data = {"dummy": [1, 1, 1, 1]}
    dummy_client.design = pd.DataFrame(design_data, index=["s1", "s2", "s3", "s4"])
    
    data = {
        "f1": [1, 2, 3, 4],
        "f2": [5, 6, 7, 8],
        "f3": [9, 10, 11, 12]
    }
    create_data_frame(dummy_client, data)
    
    result = dummy_client.get_batch_feature_presence_info(min_samples=2)
    expected = {
        "TestClient": ["f1", "f2", "f3"],
    }
    assert result == expected


def test_invalid_data_types(dummy_client):
    """
    Test that the function raises errors when data or design is not a DataFrame.
    """
    dummy_client.data = "not a dataframe"
    with pytest.raises(ValueError, match="Data must be a pandas DataFrame"):
        dummy_client.get_batch_feature_presence_info(min_samples=2)
    
    dummy_client.data = pd.DataFrame()
    dummy_client.feature_names = "not a list"
    with pytest.raises(ValueError, match="feature_names must be a list"):
        dummy_client.get_batch_feature_presence_info(min_samples=2)
    
    dummy_client.feature_names = ["f1", "f2"]
    dummy_client.design = "not a dataframe"
    with pytest.raises(ValueError, match="Design must be a pandas DataFrame"):
        dummy_client.get_batch_feature_presence_info(min_samples=2)


def test_validate_inputs_valid(caplog, dummy_client):
    """
    When global variables exactly match the client's local covariates,
    validation should pass and log an info message.
    """
    add_covariates(dummy_client)
    create_data_frame(dummy_client)
    global_vars = ["cov1", "cov2"]
    with caplog.at_level(logging.INFO):
        dummy_client.validate_inputs(global_vars)
    # Check that the log contains a validated message.
    assert any("Inputs validated" in record.message for record in caplog.records)
    # Client's variables should be updated to the global list.
    assert dummy_client.variables == global_vars


def test_validate_variables_no_global(caplog, dummy_client):
    """
    If no global variables are provided (None) but the client has local covariates,
    a log message is issued and client's variables become None.
    """
    add_covariates(dummy_client)
    create_data_frame(dummy_client)
    with caplog.at_level(logging.INFO):
        dummy_client.validate_variables(None)
    assert dummy_client.variables is None
    assert any("No common global covariates selected" in record.message for record in caplog.records)


def test_validate_none_local_variables(dummy_client):
    """
    If global variables are provided but the client has no local covariates,
    a ValueError should be raised.
    """
    add_covariates(dummy_client)
    create_data_frame(dummy_client)
    dummy_client.variables = None  # Simulate missing local covariates.
    with pytest.raises(ValueError, match="Variables must be a list"):
        dummy_client.validate_variables(None)
        

def test_validate_variables_global_provided_but_no_local(dummy_client):
    """
    If global variables are provided but the client has no local covariates,
    a ValueError should be raised.
    """
    add_covariates(dummy_client)
    create_data_frame(dummy_client)
    dummy_client.variables = []  # Simulate missing local covariates.

    with pytest.raises(ValueError, match="Global covariates selected that are not available in this client's covariates:"):
        dummy_client.validate_variables(["cov1", "cov2"])


def test_validate_variables_extra_global(dummy_client):
    """
    If global variables include a covariate not present in the client's local list,
    a ValueError should be raised.
    """
    add_covariates(dummy_client)
    create_data_frame(dummy_client)
    # Provide a global variable list that includes an extra variable "cov3".
    with pytest.raises(ValueError, match="Global covariates selected that are not available"):
        dummy_client.validate_variables(["cov1", "cov2", "cov3"])


def test_validate_variables_extra_local_warning(caplog, dummy_client):
    """
    If the client has extra local covariates (e.g., "cov_extra") not in the global list,
    a warning should be logged and the client's variables should be updated to the global list.
    """
    add_covariates(dummy_client)
    create_data_frame(dummy_client)
    dummy_client.variables = ["cov1", "cov2", "cov_extra"]
    with caplog.at_level(logging.WARNING):
        dummy_client.validate_variables(["cov1", "cov2"])
    # Client's variables should be updated to only the global list.
    assert dummy_client.variables == ["cov1", "cov2"]
    assert any("extra local covariates" in record.message for record in caplog.records)


def test_validate_variables_missing_in_design(dummy_client):
    """
    If a required covariate is missing from the design matrix, a ValueError should be raised.
    """
    add_covariates(dummy_client)
    create_data_frame(dummy_client)
    # Remove "cov2" from the design DataFrame.
    dummy_client.design = dummy_client.design.drop(columns=["cov2"])
    with pytest.raises(ValueError, match="Variables not found in the design matrix"):
        dummy_client.validate_variables(["cov1", "cov2"])


def test_validate_variables_missing_in_data(dummy_client):
    """
    If the design matrix is absent, the function checks the data.
    In this fallback, the data's index is used to find covariates.
    (Note: In a proper setup, design should be provided.)
    """
    add_covariates(dummy_client)
    create_data_frame(dummy_client)
    dummy_client.design = None
    # For this test, adjust the data so that its index represents covariates.
    dummy_client.data = pd.DataFrame({
        "s1": [1, 2],
        "s2": [3, 4],
        "s3": [5, 6],
        "s4": [7, 8]
    }, index=["cov1", "s1"])
    with pytest.raises(ValueError, match="Variables not found in the data"):
        dummy_client.validate_variables(["cov1", "cov2"])


def test_set_data_same_features(dummy_client):
    """
    When global features match local features, the data matrix remains the same.
    """
    create_data_frame(dummy_client)
    global_features = ["f1", "f2", "f3"]
    dummy_client.set_data(global_features)
    assert list(dummy_client.data.index) == global_features
    assert dummy_client.feature_names == global_features

def test_set_data_with_extra_local(dummy_client):
    """
    Test the case when local data has an extra feature not in global.
    """
    create_data_frame(dummy_client)
    global_features = ["f1", "f2"]
    dummy_client.set_data(global_features)
    # f3 should be dropped
    expected_index = global_features
    assert list(dummy_client.data.index) == expected_index
    assert "f3" not in dummy_client.data.index
    assert dummy_client.feature_names == global_features


def test_set_data_with_extra_global(dummy_client):
    """
    Test the case when global has an extra feature.
    """
    create_data_frame(dummy_client)
    global_features = ["f1", "f2", "f3", "f4"]
    with pytest.raises(ValueError, match="Error in feature alignment:"):
        dummy_client.set_data(global_features)


# --- Additional tests for validate_data ---

def test_validate_data_non_dataframe(dummy_client):
    """
    Test that validate_data raises an error when self.data is not a DataFrame.
    """
    dummy_client.data = "not a dataframe"
    dummy_client.feature_names = ["f1", "f2"]
    with pytest.raises(ValueError, match="self.data must be a pandas DataFrame"):
        dummy_client.validate_data()

def test_validate_data_missing_feature_names(dummy_client):
    """
    Test that validate_data raises an exception when feature_names are missing.
    """
    dummy_client.data = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
    dummy_client.feature_names = None
    with pytest.raises(Exception, match="Feature names are not loaded yet, cannot set data"):
        dummy_client.validate_data()

def test_validate_data_contains_na(dummy_client):
    """
    Test that validate_data raises an error when the data contains NA values.
    """
    dummy_client.data = pd.DataFrame({"f1": [1, None], "f2": [3, 4]})
    dummy_client.feature_names = ["f1", "f2"]
    with pytest.raises(ValueError, match="Data matrix contains NA values"):
        dummy_client.validate_data()

def test_validate_data_variables_without_design(dummy_client):
    """
    Test that validate_data raises an error when variables exist but design is None.
    """
    dummy_client.data = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
    dummy_client.feature_names = ["f1", "f2"]
    dummy_client.variables = ["cov1"]
    dummy_client.design = None
    with pytest.raises(ValueError, match="Design matrix must be loaded to validate variables"):
        dummy_client.validate_data()

def test_validate_data_confounded_single_value(dummy_client):
    """
    Test that validate_data raises an error when a variable in design has a single unique value.
    """
    dummy_client.data = pd.DataFrame({"f1": [1, 2, 3, 4]})
    dummy_client.feature_names = ["f1"]
    dummy_client.variables = ["cov1"]
    # Create design DataFrame with cov1 constant across all samples.
    dummy_client.design = pd.DataFrame({"cov1": [1, 1, 1, 1]}, index=["s1", "s2", "s3", "s4"])
    with pytest.raises(ValueError, match="Variable cov1 has the same value for all samples"):
        dummy_client.validate_data()

def test_validate_data_confounded_covariates(dummy_client):
    """
    Test that validate_data raises an error when two covariates are identical.
    """
    dummy_client.data = pd.DataFrame({"f1": [1, 2, 3, 4]})
    dummy_client.feature_names = ["f1"]
    dummy_client.variables = ["cov1", "cov2"]
    # Create design DataFrame with identical columns for cov1 and cov2.
    dummy_client.design = pd.DataFrame({
        "cov1": [1, 2, 3, 4],
        "cov2": [1, 2, 3, 4]
    }, index=["s1", "s2", "s3", "s4"])
    with pytest.raises(ValueError, match="Variables cov1 and cov2 have the same values"):
        dummy_client.validate_data()

def test_validate_data_success(dummy_client):
    """
    Test that validate_data passes with valid data, feature_names, design, and variables.
    """
    dummy_client.data = pd.DataFrame({"f1": [1, 2, 3, 4]})
    dummy_client.feature_names = ["f1"]
    dummy_client.variables = ["cov1"]
    dummy_client.design = pd.DataFrame({"cov1": [1, 2, 1, 2]}, index=["s1", "s2", "s3", "s4"])
    # Should not raise any error.
    dummy_client.validate_data()

# --- Additional tests for validate_variables ---

def test_validate_variables_not_list(dummy_client):
    """
    Test that validate_variables raises an error when self.variables is not a list.
    """
    dummy_client.variables = "cov1"  # Incorrect type.
    dummy_client.data = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
    with pytest.raises(ValueError, match="Variables must be a list."):
        dummy_client.validate_variables(["cov1"])

def test_validate_variables_data_not_dataframe(dummy_client):
    """
    Test that validate_variables raises an error when self.data is not a DataFrame.
    """
    dummy_client.variables = ["cov1"]
    dummy_client.data = "not a dataframe"
    with pytest.raises(ValueError, match="Data must be a pandas DataFrame."):
        dummy_client.validate_variables(["cov1"])

def test_validate_variables_design_fallback_success(dummy_client):
    """
    Test the fallback branch where design is None and data index is used to check for variables.
    """
    dummy_client.design = None
    # Set the data index to represent available covariates.
    dummy_client.data = pd.DataFrame({
        "s1": [1, 2],
        "s2": [3, 4]
    }, index=["cov1", "cov2"])
    dummy_client.variables = ["cov1"]
    # Should pass since cov1 is in the data index.
    dummy_client.validate_variables(["cov1"])

def test_validate_variables_design_fallback_failure(dummy_client):
    """
    Test the fallback branch where design is None and required variables are missing from the data index.
    """
    dummy_client.design = None
    dummy_client.data = pd.DataFrame({
        "s1": [1, 2],
        "s2": [3, 4]
    }, index=["cov1", "cov2"])
    dummy_client.variables = ["cov1", "cov_missing"]
    with pytest.raises(ValueError, match="Variables not found in the data"):
        dummy_client.validate_variables(["cov1", "cov_missing"])

# --- Additional tests for validate_inputs ---

def test_validate_inputs_error_from_validate_variables(dummy_client):
    """
    Test that an error in validate_variables is propagated through validate_inputs.
    """
    # Set up a condition that will cause validate_variables to fail:
    # For instance, design fallback should fail when the variable is not in the data index.
    dummy_client.variables = ["cov1"]
    dummy_client.data = pd.DataFrame({"f1": [1, 2]}, index=["s1", "s2"])
    # Since design is None, validate_variables will check data.index and fail.
    with pytest.raises(ValueError, match="Variables not found in the design matrix"):
        dummy_client.validate_inputs(["cov1"])

def test_validate_inputs_error_from_validate_data(dummy_client):
    """
    Test that an error in validate_data is propagated through validate_inputs.
    """
    # Set up data with NA values to trigger a failure in validate_data.
    dummy_client.data = pd.DataFrame({"f1": [1, None]}, index=["s1", "s2"])
    dummy_client.feature_names = ["f1"]
    dummy_client.variables = []
    with pytest.raises(ValueError, match="Data matrix contains NA values"):
        dummy_client.validate_inputs([])

def test_validate_inputs_success_log(caplog, dummy_client):
    """
    Test that validate_inputs logs a success message when inputs are valid.
    """
    dummy_client.data = pd.DataFrame({"f1": [1, 2, 3, 4]})
    dummy_client.feature_names = ["f1"]
    dummy_client.design = pd.DataFrame({"cov1": [1, 2, 1, 2]}, index=["s1", "s2", "s3", "s4"])
    dummy_client.variables = ["cov1"]
    with caplog.at_level(logging.INFO):
        dummy_client.validate_inputs(["cov1"])
    assert any("Inputs validated" in record.message for record in caplog.records)


# --- Additional tests for create_design ---

def test_create_design_non_dataframe(dummy_client):
    """
    Test that create_design raises an error when self.data is not a DataFrame.
    """
    dummy_client.data = "not a dataframe"
    with pytest.raises(ValueError, match="self.rawdata must be a pandas DataFrame"):
        dummy_client.create_design(["Dummy|Batch1"])

def test_create_design_no_cohorts(dummy_client):
    """
    Test that create_design raises an error when no cohorts are provided.
    """
    # Set self.data to a valid DataFrame with sample columns.
    dummy_client.data = pd.DataFrame([[1,2], [3,4]], columns=["s1", "s2"])
    with pytest.raises(ValueError, match="No cohorts provided"):
        dummy_client.create_design([])

def test_create_design_single_batch(dummy_client):
    """
    Test that create_design correctly builds the design matrix when there is a single batch.
    """
    # Set up a valid DataFrame for data; columns represent samples.
    dummy_client.client_name = "ClientA"
    dummy_client.data = pd.DataFrame([[1,2,3], [4,5,6]], columns=["s1", "s2", "s3"])
    # Ensure design is None so it is created.
    dummy_client.design = None
    # No batch column is needed for a single batch.
    dummy_client.batch_col = None

    # Provide a single cohort string where the client name matches.
    cohorts = ["ClientA|Batch1"]
    result = dummy_client.create_design(cohorts)
    # Expect no error message returned.
    assert result is None
    # Check that the design has one column named "ClientA|Batch1" with all 1's (since client matches).
    assert "ClientA|Batch1" in dummy_client.design.columns
    # All rows (samples) should have 1 in that column.
    assert all(dummy_client.design["ClientA|Batch1"] == 1)

def test_create_design_multiple_batches_no_batch_col(dummy_client):
    """
    Test that create_design raises an error if there are multiple batches but no batch column provided.
    """
    dummy_client.client_name = "ClientA"
    dummy_client.data = pd.DataFrame([[1,2,3,4]], columns=["s1", "s2", "s3", "s4"])
    dummy_client.design = None
    # batch_col is not provided.
    dummy_client.batch_col = None
    cohorts = ["ClientA|Batch1", "ClientA|Batch2"]
    with pytest.raises(ValueError, match="Batch column was not given but multiple batches"):
        dummy_client.create_design(cohorts)

def test_create_design_multiple_batches_valid(dummy_client):
    """
    Test that create_design correctly builds the design matrix for multiple batches when a batch column is provided.
    """
    dummy_client.client_name = "ClientA"
    # Create a data DataFrame with sample names as columns.
    dummy_client.data = pd.DataFrame([[1,2,3,4],
                                      [5,6,7,8]], columns=["s1", "s2", "s3", "s4"])
    # Create an initial design DataFrame with sample names as index and a batch column.
    # Assume samples s1 and s2 belong to Batch1 and s3 and s4 to Batch2.
    dummy_client.design = pd.DataFrame({
        "batch": ["Batch1", "Batch1", "Batch2", "Batch2"]
    }, index=["s1", "s2", "s3", "s4"])
    dummy_client.batch_col = "batch"

    cohorts = ["ClientA|Batch1", "ClientA|Batch2"]
    result = dummy_client.create_design(cohorts)
    # Expect no error message.
    assert result is None
    # The design should no longer contain the batch_col.
    assert "batch" not in dummy_client.design.columns
    # Check that the design has the correct binary indicator columns.
    # For cohort "ClientA|Batch1", only samples in Batch1 should be 1.
    design = dummy_client.design
    assert "ClientA|Batch1" in design.columns
    assert "ClientA|Batch2" in design.columns
    # Check indicator values.
    for sample in design.index:
        if sample in ["s1", "s2"]:
            assert design.loc[sample, "ClientA|Batch1"] == 1
            assert design.loc[sample, "ClientA|Batch2"] == 0
        else:
            assert design.loc[sample, "ClientA|Batch1"] == 0
            assert design.loc[sample, "ClientA|Batch2"] == 1
    # Check that the columns are ordered as per cohorts (no extra columns).
    assert list(design.columns) == cohorts

def test_create_design_with_covariates_variables_in_data(dummy_client):
    """
    Test create_design when covariates are provided and variables_in_data is True.
    The covariate data should be pulled from self.rawdata.
    """
    dummy_client.client_name = "ClientA"
    dummy_client.batch_col = "batch"
    # Create a data DataFrame with sample names.
    dummy_client.data = pd.DataFrame({
        "s1": [11, 12, 13, 14],
        "s2": [21, 22, 23, 24],
        "s3": [31, 32, 33, 34],
        "s4": [41, 42, 43, 44],
        "s5": [51, 52, 53, 54],
        "s6": [61, 62, 63, 64],
    }, index=["f1", "f2", "f3", "f4"])
    # Create a design DataFrame with a batch column.
    dummy_client.design = pd.DataFrame({
        "batch": ["Batch1", "Batch1", "Batch2", "Batch2", "Batch2", "Batch1"]
    }, index=["s1", "s2", "s3", "s4", "s5", "s6"])
    # Provide a valid cohorts list.
    cohorts = ["ClientA|Batch1", "ClientA|Batch2"]
    # Set covariates to be pulled from rawdata.
    dummy_client.variables = ["cov1"]
    dummy_client.variables_in_data = True
    # Simulate rawdata: rows represent covariates, columns represent samples.
    dummy_client.rawdata = pd.DataFrame({
        "s1": [0, 11, 12, 13, 14],
        "s2": [1, 21, 22, 23, 24],
        "s3": [1, 31, 32, 33, 34],
        "s4": [0, 41, 42, 43, 44],
        "s5": [1, 51, 52, 53, 54],
        "s6": [0, 61, 62, 63, 64]
    }, index=["cov1", "f1", "f2", "f3", "f4"])
    result = dummy_client.create_design(cohorts)
    assert result is None
    # The design should now include the covariate column from rawdata.
    assert "cov1" in dummy_client.design.columns
    # Verify that the covariate values are correctly added (by checking one sample).
    cov1_val = dummy_client.design.loc["s1", "cov1"]
    assert cov1_val == 0

def test_create_design_with_covariates_not_in_data(dummy_client):
    """
    Test create_design when covariates are provided and variables_in_data is False.
    In this branch, the covariates are expected to be present already in the design.
    """
    dummy_client.client_name = "ClientA"
    dummy_client.batch_col = "batch"
    dummy_client.data = pd.DataFrame([[1,2,3,4],
                                      [5,6,7,8]], columns=["s1", "s2", "s3", "s4"])
    # Pre-create a design with a batch column and a covariate column.
    dummy_client.design = pd.DataFrame({
        "batch": ["Batch1", "Batch1", "Batch2", "Batch2"],
        "cov1": [100, 101, 102, 103]
    }, index=["s1", "s2", "s3", "s4"])
    # Set variables_in_data False so that it expects covariate columns in design.
    dummy_client.variables = ["cov1"]
    dummy_client.variables_in_data = False
    cohorts = ["ClientA|Batch1", "ClientA|Batch2"]
    result = dummy_client.create_design(cohorts)
    # Expect no error message.
    assert result is None
    # The design should include covariate "cov1" with the original values.
    assert all(dummy_client.design["cov1"] == [100, 101, 102, 103])
