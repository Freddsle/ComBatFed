import unittest
import numpy as np
import pandas as pd
import logging
from classes.client import Client


class TestClientFunctions(unittest.TestCase):
    def setUp(self):
        # Disable logging to keep test output clean.
        logging.disable(logging.CRITICAL)
        # Create a Client instance.
        self.client = Client()
        
        # ---------------------------
        # Setup for get_naive_estimates() test
        # ---------------------------
        # Create a dummy design matrix (rows are sample IDs)
        design_data = {
            "GSE129508|0": [0, 0, 0, 0, 0, 0],
            "GSE149276|1": [1, 1, 1, 1, 1, 1],
            "GSE58135|2": [0, 0, 0, 0, 0, 0],
            "lum":         [0, 0, 0, 1, 1, 1]
        }
        sample_ids_naive = [
            "GSM4495401", "GSM4495424", "GSM4495417",
            "GSM4495407", "GSM4495391", "GSM4495405"
        ]
        self.client.design = pd.DataFrame(design_data, index=sample_ids_naive)
        
        # Only one batch is used for naive estimates.
        self.client.batch_labels = ['GSE149276|1']
        
        # Setup the standardized data matrix (features x samples)
        stand_data = {
            "GSM4495401": [0.1642658,  0.0356170, -3.8480295,  2.2235793,  0.6983018],
            "GSM4495424": [-0.2654076, -1.4234188, -0.9116663,  1.4109637,  0.9889557],
            "GSM4495417": [-2.09117779,-1.42341876,  0.01248283, -2.01732622,  1.52221700],
            "GSM4495407": [-1.3816034, -1.6589365, -1.5016315,  0.1154623,  0.2486924],
            "GSM4495391": [-1.3816034, -1.6589365, -1.3414492,  2.2874919,  0.3156597],
            "GSM4495405": [-1.3816034, -1.6589365,  0.7659110,  0.3123032,  3.2375598]
        }
        features_naive = ["RP1-209B5.2", "RP1-40G4P.1", "TTC7A", "TAL2", "BTLA"]
        self.client.stand_data = pd.DataFrame(stand_data, index=features_naive)
        
        # Set mean_only to False so that variance is computed.
        self.client.mean_only = False

    def test_get_naive_estimates(self):
        # Call the function to compute naive estimates.
        self.client.get_naive_estimates()
        
        # Expected gamma_hat (shape: 1 x n_features)
        expected_gamma_hat = np.array([[
            -1.056188,  # RP1-209B5.2
            -1.298005,  # RP1-40G4P.1
            -1.137397,  # TTC7A
             0.722079,  # TAL2
             1.168564   # BTLA
        ]])
        
        # Expected delta_hat (shape: 1 x n_features)
        expected_delta_hat = np.array([[
            0.7007461,  # RP1-209B5.2
            0.4401639,  # RP1-40G4P.1
            2.503518,   # TTC7A
            2.643965,   # TAL2
            1.246566    # BTLA
        ]])
        
        np.testing.assert_allclose(
            self.client.gamma_hat, expected_gamma_hat, rtol=1e-5, atol=1e-6,
            err_msg="Computed gamma_hat does not match expected values."
        )
        np.testing.assert_allclose(
            self.client.delta_hat, expected_delta_hat, rtol=1e-5, atol=1e-6,
            err_msg="Computed delta_hat does not match expected values."
        )

    def test_get_sigma_summary(self):
        # ---------------------------
        # Setup for get_sigma_summary() test
        # ---------------------------
        # Overwrite client.data with the new expression data.
        data_sigma = {
            "GSM1401723": [1.293032, 1.963789, 12.231787, 2.603238, 3.885481],
            "GSM1401719": [0.0,      1.230733, 11.415972, 2.333591, 5.847200],
            "GSM1401741": [1.497679, 1.901425, 11.845399, 2.475007, 2.475007],
            "GSM1401707": [0.0,      0.9764922, 11.0374256, 1.5535329, 1.9645931],
            "GSM1401708": [2.949946, 1.883965, 10.814741, 3.556248, 3.181522],
            "GSM1401681": [1.049498, 2.891197, 9.893288, 2.666541, 3.548202]
        }
        features_sigma = ["RP1-209B5.2", "RP1-40G4P.1", "TTC7A", "TAL2", "BTLA"]
        self.client.data = pd.DataFrame(data_sigma, index=features_sigma)
        
        # Overwrite client.design with the provided design matrix.
        design_data_sigma = {
            "GSE129508|0": [0, 0, 0, 0, 0, 0],
            "GSE149276|1": [0, 0, 0, 0, 0, 0],
            "GSE58135|2": [1, 1, 1, 1, 1, 1],
            "lum":         [0, 0, 0, 1, 1, 1]
        }
        sample_ids_sigma = [
            "GSM1401723", "GSM1401719", "GSM1401741",
            "GSM1401707", "GSM1401708", "GSM1401681"
        ]
        self.client.design = pd.DataFrame(design_data_sigma, index=sample_ids_sigma)
        
        # Provided B_hat matrix (shape: 4 x 5).
        B_hat = np.array([
            [3.0882952,  1.9350951, 10.6875188,  0.18161070, 4.563653],
            [0.8943204,  0.1101270,  9.9713466,  2.23569604, 6.699901],
            [1.4382593,  1.7045284, 11.6119707,  2.52186197, 3.608048],
            [-0.6131336, 0.2068102, -0.8110706,  0.01899534, -0.248761]
        ])
        # Provided ref_size vector.
        ref_size = np.array([6, 6, 6])
        
        # Call get_sigma_summary with these parameters.
        sigma_site = self.client.get_sigma_summary(B_hat, ref_size)
        
        # Expected sigma summary (one value per feature).
        expected_sigma = np.array([
            1.2232077,  # RP1-209B5.2
            0.3608794,  # RP1-40G4P.1
            0.2261620,  # TTC7A
            0.3443144,  # TAL2
            1.3978354   # BTLA
        ])
        
        np.testing.assert_allclose(
            sigma_site, expected_sigma, rtol=1e-5, atol=1e-6,
            err_msg="Computed sigma summary does not match expected values."
        )

if __name__ == '__main__':
    unittest.main()