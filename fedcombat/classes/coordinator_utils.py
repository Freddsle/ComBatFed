import numpy as np
import logging
from typing import List, Tuple, Union, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a custom type alias for clarity
FeatureBatchInfoType = Tuple[
    str,                               # client name
    Union[int, None],                  # client position for ordering
    Union[str, bool, None],            # reference batch flag or identifier (no longer used)
    Dict[str, List[str]]               # dictionary mapping "<client>|<batch>" to features
]

def _process_feature_batch_info(feature_batch_info):
    client_batches = {}
    batch_to_features = {}
    
    # Process each entry and build the mapping without handling reference batch logic.
    for client_name, position, ref_batch_flag, batch_info in feature_batch_info:
        batches = list(batch_info.keys())
        client_batches[client_name] = batches
        batch_to_features.update(batch_info)
        
    return client_batches, batch_to_features


def _determine_client_order(feature_batch_info, default_order):
    # If any client position is missing, use the default order.
    if any(position is None for _, position, _, _ in feature_batch_info):
        logger.info(f"Using the default client order: {default_order}")
        client_order = default_order.copy()
    else:
        # Otherwise, order clients based on the provided positions.
        client_order = [client for client, _, _, _ in sorted(feature_batch_info, key=lambda x: x[1])]  # type: ignore
        logger.info(f"Using specified client order: {client_order}")
    return client_order


def _build_cohorts_order(client_order, client_batches):
    cohorts_order = []
    # For each client, validate batch name format and sort alphabetically.
    for client in client_order:
        batches = client_batches.get(client, [])
        valid_batches = []
        for batch in batches:
            parts = batch.split("|")
            if len(parts) != 2:
                raise ValueError(f"Batch name incorrectly formatted: {batch}")
            valid_batches.append(batch)
        valid_batches.sort()
        cohorts_order.extend(valid_batches)
    logger.info(f"Cohorts order: {cohorts_order}")
    return cohorts_order


def _populate_feature_presence_matrix(global_feature_names, cohorts_order, batch_to_features):
    feature2index = {feature: idx for idx, feature in enumerate(global_feature_names)}
    num_features = len(global_feature_names)
    num_cohorts = len(cohorts_order)
    matrix = np.zeros((num_features, num_cohorts), dtype=int)

    # Fill the matrix indicating the presence of features in each batch.
    for cohort_idx, batch_name in enumerate(cohorts_order):
        if batch_name not in batch_to_features:
            raise ValueError(f"No features found for batch {batch_name}")
        batch_features = batch_to_features[batch_name]
        for feature in batch_features:
            if feature in feature2index:
                matrix[feature2index[feature], cohort_idx] = 1
    return matrix


def create_feature_presence_matrix(
        feature_batch_info: List[FeatureBatchInfoType],
        global_feature_names: List[str],
        default_order: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
    # Build the necessary mappings from the input.
    client_batches, batch_to_features = _process_feature_batch_info(feature_batch_info)
    # Determine client order based on available position info or default_order.
    client_order = _determine_client_order(feature_batch_info, default_order)
    # Build the cohorts order by sorting each client's batches.
    cohorts_order = _build_cohorts_order(client_order, client_batches)
    # Create and populate the feature presence matrix.
    matrix = _populate_feature_presence_matrix(global_feature_names, cohorts_order, batch_to_features)
    return matrix, cohorts_order


def _calculate_feature_counts(feature_batch_info: List[FeatureBatchInfoType]) -> Dict[str, int]:
    feature_count = {}
    # Count each feature's presence per client (unique per client).
    for client_name, _, _, batch_info in feature_batch_info:
        client_feature_names = set()
        for features in batch_info.values():
            for feature in features:
                if feature not in client_feature_names:
                    feature_count[feature] = feature_count.get(feature, 0) + 1
                    client_feature_names.add(feature)
    return feature_count


def select_common_features_variables(
    feature_batch_info: List[FeatureBatchInfoType],
    default_order: List[str],
    min_clients: int = 3
) -> Tuple[List[str], np.ndarray, List[str]]:
    feature_count = _calculate_feature_counts(feature_batch_info)
    # Select and sort features that meet the minimum client threshold.
    global_feature_names = sorted([feature for feature, count in feature_count.items() if count >= min_clients])
    logger.info(f"Found {len(global_feature_names)} features present in (at least) {min_clients} clients")
    logger.info(f"Total number of unique features: {len(feature_count.keys())}")

    feature_presence_matrix, cohorts_order = create_feature_presence_matrix(
        feature_batch_info, global_feature_names, default_order
    )
    return global_feature_names, feature_presence_matrix, cohorts_order
