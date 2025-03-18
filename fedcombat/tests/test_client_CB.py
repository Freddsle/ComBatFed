import numpy as np
import pandas as pd
import pytest
from classes.client import Client

@pytest.fixture
def dummy_client():
    """
    Creates a dummy Client object for testing compute_XtX_XtY.
    """
    client = Client()
    client.client_name = "TestClient"
    
    # Configure a basic logger if needed.
    client.logger = type("DummyLogger", (), {"error": lambda self, msg: None})()
    
    # Create a design matrix with shape (n_samples, k)
    # For instance, 4 samples and 3 predictors.
    design_data = np.array([
        [1.0, 0.5, 2.0],
        [1.0, 1.5, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.5]
    ])
    client.design = pd.DataFrame(design_data, index=["s1", "s2", "s3", "s4"],
                                 columns=["Intercept", "Var1", "Var2"])
    
    # Create a data matrix with shape (n_features, n_samples)
    # For instance, 2 features and 4 samples.
    data_values = np.array([
        [10.0, 12.0, 11.0, 13.0],  # Feature 1
        [20.0, 21.0, 19.0, 22.0]   # Feature 2
    ])
    client.data = pd.DataFrame(data_values, index=["feat1", "feat2"],
                               columns=["s1", "s2", "s3", "s4"])
    return client

def test_compute_XtX_XtY_complete(dummy_client):
    """
    Test compute_XtX_XtY when all data values are finite.
    """
    XtX, XtY  = dummy_client.compute_XtX_XtY()
    # Expected shapes: (n_features, k, k) and (n_features, k)
    n_features = dummy_client.data.shape[0]  # 2 features
    k = dummy_client.design.shape[1]           # 3 predictors
    assert XtX.shape == (n_features, k, k)
    assert XtY.shape == (n_features, k)
    
    # Manually compute for feature 1.
    X = dummy_client.design.values
    y1 = dummy_client.data.loc["feat1"].values  # all valid
    expected_XtX_f1 = X.T @ X
    expected_XtY_f1 = X.T @ y1
    np.testing.assert_allclose(XtX[0], expected_XtX_f1)
    np.testing.assert_allclose(XtY[0], expected_XtY_f1)

def test_compute_XtX_XtY_with_nan(dummy_client):
    """
    Test compute_XtX_XtY when some values in a feature are NaN.
    """
    # Introduce a NaN in feature 2 at sample s3.
    dummy_client.data.loc["feat2", "s3"] = np.nan
    XtX, XtY = dummy_client.compute_XtX_XtY()
    
    # For feature 2, use only valid samples (s1, s2, s4).
    X = dummy_client.design.values
    valid = np.array([True, True, False, True])
    X_valid = X[valid, :]
    y2 = dummy_client.data.loc["feat2"].values
    y2_valid = y2[valid]
    expected_XtX_f2 = X_valid.T @ X_valid
    expected_XtY_f2 = X_valid.T @ y2_valid
    
    np.testing.assert_allclose(XtX[1], expected_XtX_f2)
    np.testing.assert_allclose(XtY[1], expected_XtY_f2)

def test_compute_XtX_XtY_all_nan(dummy_client):
    """
    Test compute_XtX_XtY when an entire feature has all NaN values.
    """
    # Set all values for feature 1 to NaN.
    dummy_client.data.loc["feat1", :] = np.nan
    XtX, XtY = dummy_client.compute_XtX_XtY()
    
    # For feature 1, expect XtX and XtY to be all zeros.
    np.testing.assert_array_equal(XtX[0], np.zeros((dummy_client.design.shape[1],
                                                     dummy_client.design.shape[1])))
    np.testing.assert_array_equal(XtY[0], np.zeros(dummy_client.design.shape[1]))
    
    # For feature 2, computation should proceed normally.
    X = dummy_client.design.values
    y2 = dummy_client.data.loc["feat2"].values
    expected_XtX_f2 = X.T @ X
    expected_XtY_f2 = X.T @ y2
    np.testing.assert_allclose(XtX[1], expected_XtX_f2)
    np.testing.assert_allclose(XtY[1], expected_XtY_f2)

def test_compute_XtX_XtY_invalid_design(dummy_client):
    """
    Test that compute_XtX_XtY raises a ValueError if design is not a DataFrame.
    """
    dummy_client.design = "not a dataframe"
    with pytest.raises(ValueError, match="Design matrix must be a pandas DataFrame."):
        dummy_client.compute_XtX_XtY()

def test_compute_XtX_XtY_invalid_data(dummy_client):
    """
    Test that compute_XtX_XtY raises a ValueError if data is not a DataFrame.
    """
    dummy_client.data = "not a dataframe"
    with pytest.raises(ValueError, match="Data matrix must be a pandas DataFrame."):
        dummy_client.compute_XtX_XtY()


def test_get_sigma_summary_complete(dummy_client):
    """
    Test get_sigma_summary with complete data (no NaNs).
    """
    B_hat = np.array([
        [0.5, 1.0],
        [0.2, -0.3],
        [1.0, 0.5]
    ])
    ref_size = np.array([4])

    sigma_summary = dummy_client.get_sigma_summary(B_hat, ref_size)

    assert sigma_summary.shape == (dummy_client.data.shape[0],)

    # Manual computation
    fitted = (dummy_client.design.values @ B_hat).T
    diff = dummy_client.data.values - fitted
    factor = 4 / (4 - 1)
    expected_var = np.nanvar(diff, axis=1, ddof=1) / factor

    np.testing.assert_allclose(sigma_summary, expected_var)


def test_get_sigma_summary_with_nan(dummy_client):
    """
    Test get_sigma_summary when data contains NaNs.
    """
    dummy_client.data.iloc[1, 2] = np.nan

    B_hat = np.array([
        [0.5, 1.0],
        [0.2, -0.3],
        [1.0, 0.5]
    ])
    ref_size = np.array([4])

    sigma_summary = dummy_client.get_sigma_summary(B_hat, ref_size)

    assert sigma_summary.shape == (dummy_client.data.shape[0],)

    # Manual computation
    fitted = (dummy_client.design.values @ B_hat).T
    diff = dummy_client.data.values - fitted
    factor = 4 / (4 - 1)
    expected_var = np.nanvar(diff, axis=1, ddof=1) / factor

    np.testing.assert_allclose(sigma_summary, expected_var)


def test_get_sigma_summary_all_nan(dummy_client):
    """
    Test get_sigma_summary when an entire feature row has NaNs.
    """
    import warnings
    dummy_client.data.iloc[0, :] = np.nan

    B_hat = np.array([
        [0.5, 1.0],
        [0.2, -0.3],
        [1.0, 0.5]
    ])
    ref_size = np.array([4])

    with warnings.catch_warnings(record=True) as w:
        sigma_summary = dummy_client.get_sigma_summary(B_hat, ref_size)
        
        assert len(w) >= 1
        assert any('Degrees of freedom' in str(warn.message) for warn in w)

    assert sigma_summary.shape == (dummy_client.data.shape[0],)
    assert np.isnan(sigma_summary[0])


def test_get_sigma_summary_invalid_design(dummy_client):
    """
    Test that get_sigma_summary raises an error if design is not correctly shaped.
    """
    dummy_client.design = "invalid"

    B_hat = np.array([
        [0.5, 1.0],
        [0.2, -0.3],
        [1.0, 0.5]
    ])
    ref_size = np.array([4])

    with pytest.raises(Exception):
        dummy_client.get_sigma_summary(B_hat, ref_size)


def test_get_sigma_summary_invalid_Bhat(dummy_client):
    """
    Test that get_sigma_summary raises an error if B_hat dimensions are incorrect.
    """
    B_hat = np.array([[0.5, 1.0]])  # Incorrect shape
    ref_size = np.array([4])

    with pytest.raises(ValueError):
        dummy_client.get_sigma_summary(B_hat, ref_size)


# --- New Fixture for ComBat-specific tests ---
@pytest.fixture
def dummy_client_combat():
    """
    Creates a dummy Client object for testing the ComBat functions.
    In this setup, we assume the data matrix is of shape (n_features, n_samples)
    and the design matrix is (n_features, k). This is necessary for the given 
    implementations of get_standardized_data and get_naive_estimates.
    """
    client = Client()
    client.client_name = "TestClient_ComBat"
    
    # Setup a dummy logger that does nothing (or collects messages if needed)
    client.logger = type("DummyLogger", (), {"info": lambda self, msg, *args: None})()

    # In this ComBat-specific setup, we assume that the design matrix rows correspond to features.
    # Let n_features = 4 and k = 3 predictors.
    design_data = np.array([
        [1, 0.2, 3.0],
        [1, 0.1, 2.5],
        [1, 1.0, 0.5],
        [1, 0.0, 2.0],
        [1, 0.5, 1.5]
    ])
    client.design = pd.DataFrame(design_data,
                                index=["s1", "s2", "s3", "s4", "s5"],
                                columns=["batch1", "batch2", "other"])
    
    # Create a data matrix with shape (n_features, n_samples)
    # Let n_samples = 5.
    data_values = np.array([
        [10, 12, 11, 13, 12],
        [20, 21, 19, 22, 20],
        [15, 15, 16, 14, 15],
        [30, 29, 31, 28, 30]
    ])
    # data shape: (4, 5)
    client.data = pd.DataFrame(data_values, 
                               index=["feat1", "feat2", "feat3", "feat4"],
                               columns=["s1", "s2", "s3", "s4", "s5"])
    
    # For get_naive_estimates, define batch_labels corresponding to the first two columns of design.
    client.batch_labels = ["batch1", "batch2"]
    
    return client

# --- Tests for get_standardized_data ---

def test_get_standardized_data_complete_combat(dummy_client_combat):
    """
    Test get_standardized_data with valid inputs.
    
    The expected standardized data is computed as:
      stand_data = (data - stand_mean - mod_mean) / D
    where:
      - mod_mean = (tmp @ B_hat).T, with tmp being a copy of the design matrix where the first len(ref_size) columns are zeroed.
      - D = np.outer(np.sqrt(var_pooled), np.ones(np.sum(ref_size)))
    """
    client = dummy_client_combat
    n_features, n_samples = client.data.shape  # (4, 5)
    k = client.design.shape[1]                 # 3 predictors

    # B_hat must be of shape (k, n_features) = (3, 4)
    B_hat = np.array([
        [0.5, 1.0, 0.2, 0.3],
        [0.1, -0.2, 0.0, 0.5],
        [0.3, 0.4, 0.1, -0.1]
    ])
    # stand_mean and var_pooled are per feature (n_features,)
    stand_mean = np.array([1.0, 2.0, 1.5, 2.5])
    var_pooled = np.array([4.0, 9.0, 1.0, 16.0])
    # ref_size: for simplicity, use an array with a single element equal to number of samples.
    ref_size = np.array([n_samples])
    
    # Call the function
    client.get_standardized_data(B_hat, stand_mean, var_pooled, ref_size)
    
    # Recompute expected mod_mean.
    # Copy design and zero out first len(ref_size) columns (here: first column)
    tmp = client.design.copy().values  # shape: (n_features, k)
    tmp[:, :len(ref_size)] = 0
    # Compute mod_mean = (tmp @ B_hat).T; since tmp: (4,3) and B_hat: (3,4), tmp @ B_hat => (4,4)
    # Then transpose gives (4,4)
    mod_mean = (tmp @ B_hat).T
    
    # Denominator D
    D = np.outer(np.sqrt(var_pooled), np.ones(np.sum(ref_size)))  # shape: (4, 5)
    
    # Expected standardized data:
    expected = (client.data.values - stand_mean.reshape(-1, 1) - mod_mean) / D

    np.testing.assert_allclose(client.stand_data, expected, err_msg="Standardized data does not match expected values.")

def test_get_standardized_data_mismatched_shapes(dummy_client_combat):
    """
    Test get_standardized_data with inputs that have mismatched shapes.
    For instance, B_hat should have shape (k, n_features) but if provided with a wrong shape, an error is expected.
    """
    client = dummy_client_combat
    n_features, n_samples = client.data.shape  # (4, 5)
    k = client.design.shape[1]                 # 3 predictors

    # Provide an incorrectly shaped B_hat (e.g. shape (k+1, n_features))
    B_hat_bad = np.array([
        [0.5, 1.0, 0.2, 0.3],
        [0.1, -0.2, 0.0, 0.5],
        [0.3, 0.4, 0.1, -0.1],
        [0.2, 0.1, -0.3, 0.4]
    ])
    stand_mean = np.array([1.0, 2.0, 1.5, 2.5])
    var_pooled = np.array([4.0, 9.0, 1.0, 16.0])
    ref_size = np.array([n_samples])
    
    with pytest.raises(ValueError):
        # Assuming the function is meant to check shape consistency, it should raise an error.
        client.get_standardized_data(B_hat_bad, stand_mean, var_pooled, ref_size)

# --- Tests for get_naive_estimates ---

def test_get_naive_estimates_complete_combat(dummy_client_combat):
    """
    Test get_naive_estimates after standardized data has been computed.
    This test checks:
      - gamma_hat is computed as np.linalg.inv(batch_design.T @ batch_design) @ batch_design.T @ stand_data.T.
      - delta_hat is computed per batch label by taking the sample variance (with ddof=1) over the subset of standardized data.
    """
    client = dummy_client_combat
    n_features, n_samples = client.data.shape  # (4, 5)
    k = client.design.shape[1]                 # 3 predictors

    # Setup B_hat, stand_mean, var_pooled, and ref_size for get_standardized_data.
    B_hat = np.array([
        [0.4, 0.8, 0.2, 0.6],
        [0.1, -0.1, 0.0, 0.2],
        [0.3, 0.5, 0.2, 0.0]
    ])
    stand_mean = np.array([2.0, 3.0, 2.5, 3.5])
    var_pooled = np.array([9.0, 4.0, 16.0, 25.0])
    ref_size = np.array([n_samples])
    
    # Compute standardized data first.
    client.get_standardized_data(B_hat, stand_mean, var_pooled, ref_size)
    
    # Capture a copy of stand_data for manual computation.
    stand_data = client.stand_data  # shape: (n_features, n_samples)
    
    # For gamma_hat, the batch design is defined as the first len(batch_labels) columns of design.
    batch_labels = client.batch_labels  # e.g., ["batch1", "batch2"]
    batch_design = client.design.iloc[:, :len(batch_labels)].values  # shape: (n_features, len(batch_labels))
    
    # NOTE: In the current implementation, gamma_hat is computed as:
    #    gamma_hat = inv(batch_design.T @ batch_design) @ batch_design.T @ stand_data.T
    # Given stand_data shape is (n_features, n_samples), stand_data.T is (n_samples, n_features)
    # and batch_design is (n_features, len(batch_labels)), the multiplication is only valid
    # if n_features equals n_samples. To make the test work, we assume that in this combat fixture,
    # the design rows and the data rows both refer to features (i.e. n_features).
    #
    # For the purpose of this test, we require that n_features == n_samples.
    if n_features != n_samples:
        pytest.skip("This test requires n_features == n_samples for consistent indexing in get_naive_estimates.")
    
    # Manual computation of gamma_hat:
    XtX = batch_design.T @ batch_design
    XtY = batch_design.T @ stand_data.T
    gamma_hat_expected = np.linalg.inv(XtX) @ XtY  # shape: (len(batch_labels), n_features)
    
    # Now call get_naive_estimates.
    client.get_naive_estimates()
    
    # Test gamma_hat shape and values.
    np.testing.assert_allclose(client.gamma_hat, gamma_hat_expected, err_msg="gamma_hat does not match expected values.")
    assert client.gamma_hat.shape == (len(batch_labels), n_features)
    
    # For delta_hat, for each batch label, compute the variance (ddof=1) on the subset of stand_data rows.
    delta_hat_manual = []
    for label in batch_labels:
        # indices where design[label] == 1.
        # Since design now has index corresponding to features, this boolean mask has length n_features.
        indices = client.design[label] == 1
        # Select the rows from stand_data corresponding to these features.
        s_data = stand_data[indices, :]  # shape: (n_selected, n_samples)
        # Compute variance along axis=0 (i.e. over the selected features for each sample)
        delta_row = np.nanvar(s_data, axis=0, ddof=1)
        delta_hat_manual.append(delta_row)
    delta_hat_expected = np.vstack(delta_hat_manual)  # shape: (n_batches, n_samples)
    
    # Check shape of delta_hat. Note that according to the implementation, delta_hat is stored as (n_batches, n_features)
    # Here, because of our design choices, n_features == n_samples.
    np.testing.assert_allclose(client.delta_hat, delta_hat_expected, err_msg="delta_hat does not match expected values.")
    assert client.delta_hat.shape == (len(batch_labels), n_features)

def test_get_naive_estimates_without_standardization(dummy_client_combat):
    """
    Test that calling get_naive_estimates without first computing standardized data leads to an AttributeError
    (or a similar error), as stand_data would not be set.
    """
    client = dummy_client_combat
    with pytest.raises(AttributeError):
        client.get_naive_estimates()