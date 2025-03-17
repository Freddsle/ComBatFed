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


def aggregate_XtX_XtY(
    XtX_XtY_lists: List[List[np.ndarray]],
    n: int,
    k: int,
    use_smpc: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gets a list of a List containing the XtX and XtY matrices from each client
    and aggregates them into a single XtX_global and XtY_global matrix.
    Args:
        XtX_XtY_list: A list of tuples containing the XtX and XtY matrices from
            each client. The first element of the tuple is the XtX matrix and the
            second element is the XtY matrix.
            XtX should be of shape n x k x k and XtY should be of shape n x k.
        n: The expected number of features.
        k: The expected number of columns of the design matrix. Generally, is
            len([len(clients)] + [covariates])
            XtX should be of shape n x k x k and XtY should be of shape n x k.
        use_smpc: A boolean indicating if then computation was done using SMPC
            if yes, the data is already aggregated
    Returns:
        (XtX_global, XtY_global): A tuple containing the aggregated XtX and XtY
            matrices.
    """
    if len(XtX_XtY_lists) == 0:
        raise ValueError("No data received from clients")

    XtX_global = np.zeros((n, k, k))
    XtY_global = np.zeros((n, k))

    for XtX, XtY in XtX_XtY_lists:
        # due to serialization, the matrices are received as lists
        XtX = np.array(XtX)
        XtY = np.array(XtY)
        if XtX.shape[0] != n or XtX.shape[1] != k or XtY.shape[0] != n or XtY.shape[1] != k:
            raise ValueError(f"Shape of received XtX or XtY does not match the expected shape: {XtX.shape} {XtY.shape}")
        XtX_global += XtX
        XtY_global += XtY

    return XtX_global, XtY_global


def compute_B_hat(
    XtX_global: np.ndarray,
    XtY_global: np.ndarray
) -> np.ndarray:
    """
    Computes the B_hat matrix for the ComBat algorithm.
    Args:
        XtX_global: The aggregated XtX matrix of shape n x k x k.
        XtY_global: The aggregated XtY matrix of shape n x k.
    Returns:
        B_hat: The B_hat matrix of shape k x n.
    """
    try:
        XtX_global_inv = np.linalg.inv(XtX_global)
    except np.linalg.LinAlgError:
        raise ValueError("The XtX_global matrix is singular and cannot be inverted.")
    B_hat = XtX_global_inv @ XtY_global  # B_hat has shape (k x n)
    return B_hat

def compute_mean(
    XtX_global: np.ndarray,
    XtY_global: np.ndarray,
    B_hat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the grand mean and standardized mean for the ComBat algorithm.
    Args:
        XtX_global: The aggregated XtX matrix of shape n x k x k.
        XtY_global: The aggregated XtY matrix of shape n x k.
        B_hat: The B_hat matrix of shape k x n.
    Returns:
        grand_mean: The grand mean vector of shape n.
        stand_mean: The standardized mean matrix of shape n x k.
    """
    # Compute the grand mean:
    # Compute weighted average of the first n_batch rows of B_hat.
    # The weights are n_batches_arr / n_array.
    weights = n_batches_arr / n_array  # shape (n_batch,)
    # B_hat[0:n_batch, :] has shape (n_batch, features)
    grand_mean = weights @ B_hat[0:n_batch, :] 
    
    # Replicate grand_mean to create stand_mean, a matrix of shape (features, n_array)
    stand_mean = np.outer(grand_mean, np.ones(n_array))
    return grand_mean, stand_mean
)