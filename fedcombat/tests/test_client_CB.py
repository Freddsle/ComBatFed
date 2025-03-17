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
    XtX, XtY = dummy_client.compute_XtX_XtY()
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
