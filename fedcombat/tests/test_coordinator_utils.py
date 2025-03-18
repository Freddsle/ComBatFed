import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from classes.coordinator_utils import (
    create_feature_presence_matrix,
    select_common_features_variables,
    aggregate_XtX_XtY,
    compute_B_hat,
    compute_mean,
    get_pooled_variance,
    FeatureBatchInfoType  # if exported, otherwise, use Tuple[str, Union[int, None], Union[str, bool, None], dict]
)

class TestBatchEffectFunctions(unittest.TestCase):
    
    def setUp(self):
        # Common default order for clients
        self.default_order = ["client1", "client2", "client3"]

    def test_create_feature_presence_matrix_valid(self):
        # Define feature_batch_info with two clients
        feature_batch_info = [
            ("client1", None, False, {"client1|batch1": ["f1", "f2"]}),
            # Client2 is the reference with a single batch (flag as True)
            ("client2", None, True, {"client2|batch1": ["f2", "f3"]})
        ]
        global_feature_names = sorted(["f1", "f2", "f3"])
        matrix, cohorts_order = create_feature_presence_matrix(feature_batch_info, global_feature_names, self.default_order)
        
        # Expected cohorts_order: default_order puts client2 (the reference) at the end.
        # In this case, client1 comes first and then client2.
        expected_cohorts_order = ["client1|batch1", "client2|batch1"]
        self.assertEqual(cohorts_order, expected_cohorts_order)
        
        # Expected matrix shape is (3,2)
        self.assertEqual(matrix.shape, (3, 2))
        # Feature "f1" only in client1 batch, "f2" in both, "f3" only in client2 batch.
        expected_matrix = np.array([
            [1, 0],  # f1
            [1, 1],  # f2
            [0, 1]   # f3
        ])
        np.testing.assert_array_equal(matrix, expected_matrix)

    def test_create_feature_presence_matrix_specified_order(self):
        # All clients with explicit integer positions.
        feature_batch_info = [
            ("client1", 0, False, {"client1|batchA": ["f1", "f2"]}),
            ("client2", 2, False, {"client2|batchA": ["f2", "f3"]}),
            ("client3", 1, False, {"client3|batchA": ["f1", "f3"]})
        ]
        global_feature_names = sorted(["f1", "f2", "f3"])
        # Specified order from positions should yield order ["client1", "client2", "client3"]
        matrix, cohorts_order = create_feature_presence_matrix(feature_batch_info, global_feature_names, self.default_order)
        
        # Since client2 is reference, it should be the last in the order. Hence, expected order:
        expected_order = ["client1|batchA", "client3|batchA", "client2|batchA"]
        self.assertEqual(cohorts_order, expected_order)


    def test_select_common_features_variables(self):
        # Setup three clients with overlapping and unique features.
        feature_batch_info = [
            # Client1 has two batches (non-reference) with features f1 and f2.
            ("client1", None, False, {"client1|batch1": ["f1", "f2"], "client1|batch2": ["f1"]}),
            # Client2 is reference (single batch) with features f2 and f3.
            ("client2", None, False, {"client2|batch1": ["f2", "f3"]}),
            # Client3 (non-reference) with a single batch having features f2 and f4.
            ("client3", None, False, {"client3|batch1": ["f2", "f4"]})
        ]
        # Use a lower min_clients threshold to see union behavior
        global_features, matrix, cohorts_order = select_common_features_variables(feature_batch_info, self.default_order, min_clients=2)
        # Expected: f2 appears in all three clients, f1 only in client1, f3 only in client2, f4 only in client3.
        # For min_clients=2, only f2 should be selected.
        self.assertEqual(global_features, ["f2"])
        
        # The resulting matrix should have one row corresponding to f2.
        self.assertEqual(matrix.shape[0], 1)
        # Determine expected cohorts_order based on default ordering and reference rules.
        # client1 batches: ["client1|batch1", "client1|batch2"]
        # client2 batches: ["client2|batch1"] (reference client, so placed last)
        # client3 batches: ["client3|batch1"]
        # default_order is ["client1", "client2", "client3"], but reference client ("client2") is moved to last.
        expected_order = ["client1|batch1", "client1|batch2", "client2|batch1", "client3|batch1"]
        self.assertEqual(cohorts_order, expected_order)
        
        # For each batch, f2 should be marked as present if present in the corresponding batch.
        # client1: both batches have f2 in first batch, but not in second if "client1|batch2" only has f1.
        # client3: has no f2? Actually, client3 has f2.
        # Let's re-read: 
        #   client1: batch1 has f1 and f2, batch2 has f1 (thus, f2 not present in batch2)
        #   client2: batch1 has f2 and f3
        #   client3: batch1 has f2 and f4
        # Expected binary matrix for f2 across order: 
        # client1|batch1: 1, client1|batch2: 0, client3|batch1: 1, client2|batch1: 1
        expected_matrix = np.array([[1, 0, 1, 1]])
        np.testing.assert_array_equal(matrix, expected_matrix)

    def test_invalid_batch_format(self):
        # Test for batch name that is not correctly formatted.
        feature_batch_info = [
            ("client1", None, False, {"client1-batch1": ["f1", "f2"]}),  # Incorrect delimiter
            ("client2", None, True, {"client2|batch1": ["f2", "f3"]})
        ]
        global_feature_names = sorted(["f1", "f2", "f3"])
        with self.assertRaises(ValueError) as context:
            create_feature_presence_matrix(feature_batch_info, global_feature_names, self.default_order)
        self.assertIn("Batch name incorrectly formatted", str(context.exception))


class TestAggregateXtX_XtY(unittest.TestCase):
    
    def test_empty_input(self):
        # Test that an empty list raises a ValueError.
        with self.assertRaises(ValueError) as context:
            aggregate_XtX_XtY([], n=2, k=3)
        self.assertIn("No data received from clients", str(context.exception))
        
    def test_correct_aggregation(self):
        # Define dimensions
        n, k = 2, 3
        
        # Create two clients with valid shapes.
        # For each client, XtX is of shape (n, k, k) and XtY is of shape (n, k)
        client1_XtX = np.array([[[1, 2, 3],
                                  [4, 5, 6],
                                  [7, 8, 9]],
                                 [[1, 0, 1],
                                  [0, 1, 0],
                                  [1, 0, 1]]])
        client1_XtY = np.array([[1, 2, 3],
                                [4, 5, 6]])
        
        client2_XtX = np.array([[[9, 8, 7],
                                  [6, 5, 4],
                                  [3, 2, 1]],
                                 [[0, 1, 0],
                                  [1, 0, 1],
                                  [0, 1, 0]]])
        client2_XtY = np.array([[9, 8, 7],
                                [6, 5, 4]])
        
        # Create the list of client matrices
        XtX_XtY_lists = [
            (client1_XtX.tolist(), client1_XtY.tolist(), 5),
            (client2_XtX.tolist(), client2_XtY.tolist(), None)
        ]
        
        # Expected results computed by summing the two clients.
        expected_XtX = client1_XtX + client2_XtX
        expected_XtY = client1_XtY + client2_XtY
        
        # Run the aggregator
        XtX_global, XtY_global, _ = aggregate_XtX_XtY(XtX_XtY_lists, n=n, k=k)
        
        # Validate the results
        np.testing.assert_array_equal(XtX_global, expected_XtX)
        np.testing.assert_array_equal(XtY_global, expected_XtY)
    
    def test_invalid_shape(self):
        # Define dimensions
        n, k = 2, 3
        
        # Create one client with correct shape and one with an invalid shape.
        valid_XtX = np.zeros((n, k, k))
        valid_XtY = np.zeros((n, k))
        
        # Create an invalid client whose XtX has wrong dimensions.
        invalid_XtX = np.zeros((n + 1, k, k))  # invalid: n+1 instead of n
        invalid_XtY = np.zeros((n, k))
        
        XtX_XtY_lists = [
            (valid_XtX.tolist(), valid_XtY.tolist(), None),
            (invalid_XtX.tolist(), invalid_XtY.tolist(), 5)
        ]
        
        # Expect a ValueError indicating a shape mismatch.
        with self.assertRaises(ValueError) as context:
            aggregate_XtX_XtY(XtX_XtY_lists, n=n, k=k)
        self.assertIn("Shape of received XtX or XtY does not match the expected shape", str(context.exception))


class TestComputeBHat(unittest.TestCase):

    def test_compute_B_hat_valid(self):
        XtX_global = np.array([
            [[2, 0], [0, 2]],
            [[1, 0], [0, 1]]
        ])
        XtY_global = np.array([
            [2, 4],
            [1, 1]
        ])

        B_hat = compute_B_hat(XtX_global, XtY_global)
        expected_B_hat = np.array([
            [1, 1],
            [2, 1]
        ])

        assert_array_almost_equal(B_hat, expected_B_hat)

    def test_compute_B_hat_singular(self):
        XtX_global = np.array([
            [[1, 1], [1, 1]],  # Singular
            [[1, 0], [0, 1]]
        ])
        XtY_global = np.array([
            [2, 2],
            [1, 1]
        ])

        with self.assertRaises(ValueError) as ctx:
            compute_B_hat(XtX_global, XtY_global)
        self.assertIn("singular and cannot be inverted", str(ctx.exception))


class TestComputeMean(unittest.TestCase):

    def test_compute_mean_valid(self):
        XtX_global = np.zeros((2, 2, 2))  # Not used, placeholder
        XtY_global = np.zeros((2, 2))     # Not used, placeholder
        B_hat = np.array([
            [1, 2],
            [3, 4]
        ])
        ref_size = np.array([3, 7])

        grand_mean, stand_mean = compute_mean(XtX_global, XtY_global, B_hat, ref_size)

        expected_grand_mean = (3 * B_hat[0] + 7 * B_hat[1]) / 10
        expected_stand_mean = np.outer(expected_grand_mean, np.ones(10))

        assert_array_almost_equal(grand_mean, expected_grand_mean)
        assert_array_almost_equal(stand_mean, expected_stand_mean)


class TestGetPooledVariance(unittest.TestCase):

    def test_get_pooled_variance_valid(self):
        vars_list = [np.array([2.0, 4.0]), np.array([6.0, 8.0])]
        ref_size = np.array([10, 20])

        pooled_var = get_pooled_variance(vars_list, ref_size)
        expected_pooled_var = (10 * vars_list[0] + 20 * vars_list[1]) / 30

        assert_array_almost_equal(pooled_var, expected_pooled_var)

    def test_get_pooled_variance_single_client(self):
        vars_list = [np.array([5.0, 7.0])]
        ref_size = np.array([15])

        pooled_var = get_pooled_variance(vars_list, ref_size)
        expected_pooled_var = vars_list[0]

        assert_array_equal(pooled_var, expected_pooled_var)

    def test_get_pooled_variance_empty_vars_list(self):
        vars_list = []
        ref_size = np.array([])

        with self.assertRaises(IndexError):
            get_pooled_variance(vars_list, ref_size)


if __name__ == '__main__':
    unittest.main()

