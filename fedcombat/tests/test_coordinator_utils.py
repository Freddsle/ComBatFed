import unittest
import numpy as np
from classes.coordinator_utils import (
    create_feature_presence_matrix,
    select_common_features_variables,
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

if __name__ == '__main__':
    unittest.main()
