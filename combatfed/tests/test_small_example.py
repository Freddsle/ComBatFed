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


class TestEBEstimatorsAndCorrectedData(unittest.TestCase):
    def setUp(self):
        # Disable logging for clean test output.
        logging.disable(logging.CRITICAL)
        
        # Instantiate the client.
        self.client = Client()
        self.client.parametric = True
        self.client.eb_param = True
        self.client.mean_only = False

        # -----------------------------
        # Set pre-computed naive estimates
        # -----------------------------
        # gamma_hat and delta_hat are 1 x 5 arrays.
        self.client.gamma_hat = np.array([[1.298485,0.727352, 0.07423352, -2.129352, -0.3961784]])
        self.client.delta_hat = np.array([[0.5595267, 2.001668, 0.5465419, 0.9345964, 1.38898]])
        self.client.batch_labels = ['GSE129508|0']
        
        # -----------------------------
        # Set the design matrix for EB estimation.
        # (We need a design with one column "GSE129508|0" that flags all samples.)
        sample_ids = ["GSM3714629", "GSM3714603", "GSM3714599", 
                      "GSM3714589", "GSM3714639", "GSM3714625"]
        design_dict = {"GSE129508|0": [1]*len(sample_ids)}
        self.client.design = pd.DataFrame(design_dict, index=sample_ids)
        
        # Also, set the batch_design similarly.
        self.client.batch_design = pd.DataFrame(design_dict, index=sample_ids)
        
        # -----------------------------
        # Set the standardized data matrix (stand_data)
        # Rows: features (order: RP1-209B5.2, RP1-40G4P.1, TTC7A, TAL2, BTLA)
        # Columns: six sample IDs.
        stand_data_dict = {
            "GSM3714629": [1.1803868, -0.1721392, 1.3226065, -3.1605758, 0.8151396],
            "GSM3714603": [1.46768074, 1.09093949, -0.01425371, -3.16057578, -0.48867219],
            "GSM3714599": [2.0887994, -1.5552433, -0.5711011, -1.5123193, -2.5267857],
            "GSM3714589": [0.43260298, 1.04850659, -0.37939900, -2.14235351, -0.04613679],
            "GSM3714639": [0.4799340, 1.4124130, -0.4769703, -2.1423535, -0.6075648],
            "GSM3714625": [2.1415064, 2.5396352, 0.5645188, -0.6579344, 0.4769496]
        }
        
        features_order = ["RP1-209B5.2", "RP1-40G4P.1", "TTC7A", "TAL2", "BTLA"]
        self.client.stand_data = pd.DataFrame(stand_data_dict, index=features_order)
        
        # For get_corrected_data, self.data is used to reset the row names.
        # Here, we set self.data equal to stand_data.
        self.client.data = self.client.stand_data.copy()
        
        # -----------------------------
        # Set the mean adjustments (mod_mean and stand_mean)
        # -----------------------------
        mod_mean_dict = {
            "GSM3714629": [0, 0, 0, 0, 0],
            "GSM3714603": [0, 0, 0, 0, 0],
            "GSM3714599": [0, 0, 0, 0, 0],
            "GSM3714589": [-1.1720658, 0.1575522, -0.6811061, -0.7083514, -0.1875558],
            "GSM3714639": [-1.1720658, 0.1575522, -0.6811061, -0.7083514, -0.1875558],
            "GSM3714625": [-1.1720658, 0.1575522, -0.6811061, -0.7083514, -0.1875558]
        }
        # Transpose so that rows are features and columns are samples.
        self.client.mod_mean = pd.DataFrame(mod_mean_dict, index=features_order)
        
        stand_mean_dict = {
            "GSM3714629": [2.474147, 1.368290, 10.581513, 2.198732, 4.086554],
            "GSM3714603": [2.474147, 1.368290, 10.581513, 2.198732, 4.086554],
            "GSM3714599": [2.474147, 1.368290, 10.581513, 2.198732, 4.086554],
            "GSM3714589": [2.474147, 1.368290, 10.581513, 2.198732, 4.086554],
            "GSM3714639": [2.474147, 1.368290, 10.581513, 2.198732, 4.086554],
            "GSM3714625": [2.474147, 1.368290, 10.581513, 2.198732, 4.086554],
        }
        self.client.stand_mean = pd.DataFrame(stand_mean_dict, index=features_order)
        
        # -----------------------------
        
        self.var_pooled = np.array([ 0.8083792, 0.7740334, 0.4786854, 0.4839632, 1.3537216 ])
        
    def test_eb_estimators_and_corrected_data(self):
        # Run the EB estimator pipeline.
        self.client.get_eb_estimators()
        
        # --- Intermediate checks ---
        # Compute overall gamma_bar as mean of gamma_hat (since there's one batch, overall mean = scalar).
        computed_gamma_bar = np.nanmean(self.client.gamma_hat)
        expected_gamma_bar = -0.08509198
        self.assertAlmostEqual(computed_gamma_bar, expected_gamma_bar, places=5,
                               msg="Overall gamma_bar does not match expected value.")
        
        # Compute t2 from gamma_hat using sample variance (ddof=1) along axis=1.
        computed_t2 = self.client.gamma_hat.var(ddof=1, axis=1)[0]
        expected_t2 = 1.718877 
        self.assertAlmostEqual(computed_t2, expected_t2, places=5,
                               msg="Computed t2 does not match expected value.")
        
        # Compute a_prior and b_prior using the provided functions.
        # Wrap delta_hat in a DataFrame (as expected by apriorMat and bpriorMat).
        delta_hat_df = pd.DataFrame(self.client.delta_hat, columns=["RP1-209B5.2", "RP1-40G4P.1", "TTC7A", "TAL2", "BTLA"])
        computed_a_prior = self.client.apriorMat(delta_hat_df).iloc[0]
        computed_b_prior = self.client.bpriorMat(delta_hat_df).iloc[0]
        expected_a_prior = 5.102406 
        expected_b_prior = 4.45629
        self.assertAlmostEqual(computed_a_prior, expected_a_prior, places=5,
                               msg="Computed a_prior does not match expected value.")
        self.assertAlmostEqual(computed_b_prior, expected_b_prior, places=5,
                               msg="Computed b_prior does not match expected value.")
        
        # Check that the computed gamma_star and delta_star match expected values.
        expected_gamma_star = np.array([[1.195561, 0.6341967, 0.06250041, -1.953704, -0.3657846]])
        expected_delta_star = np.array([[0.8288581, 1.335673, 0.8198712, 0.9694374, 1.116736]])
        np.testing.assert_allclose(
            self.client.gamma_star, expected_gamma_star, rtol=1e-5, atol=1e-6,
            err_msg="Computed gamma_star does not match expected values."
        )
        np.testing.assert_allclose(
            self.client.delta_star, expected_delta_star, rtol=1e-5, atol=1e-6,
            err_msg="Computed delta_star does not match expected values."
        )
        
        # --- Final corrected data ---
        corrected_data = self.client.get_corrected_data(self.var_pooled)
        
        # The expected final corrected data as a DataFrame.
        expected_corrected_dict = {
            "GSM3714629": [2.4591613, 0.7544635, 11.5443649, 1.3460107, 5.3867578],
            "GSM3714603": [2.742884, 1.715987, 10.522865, 1.346011, 3.951254],
            "GSM3714599": [3.3562814, -0.2984304, 10.0973760, 2.5105954, 1.7072799],
            "GSM3714589": [0.5486074, 1.8412374, 9.5627502, 1.3570898, 4.2509321],
            "GSM3714639": [0.5953501, 2.1182627, 9.4881956, 1.3570898, 3.6327968],
            "GSM3714625": [2.236267, 2.976365, 10.284001, 2.405914, 4.826853]
        }
        # Expected DataFrame: rows are features in order [RP1-209B5.2, RP1-40G4P.1, TTC7A, TAL2, BTLA]
        expected_corrected = pd.DataFrame(expected_corrected_dict, index=["RP1-209B5.2", "RP1-40G4P.1", "TTC7A", "TAL2", "BTLA"])
        
        # Assert that the final corrected data matches the expected output.
        pd.testing.assert_frame_equal(
            corrected_data, expected_corrected,
            obj="Final corrected data does not match expected values."
        )


if __name__ == '__main__':
    unittest.main()