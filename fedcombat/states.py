import os
import numpy as np

from typing import List, Dict
import logging

from FeatureCloud.app.engine.app import AppState, app_state, Role, LogLevel

from classes.client import Client
import classes.coordinator_utils as c_utils

INPUT_FOLDER = os.path.join("mnt", "input")

logging.basicConfig(level=logging.INFO)

@app_state('initial')
class InitialState(AppState):

    def register(self):
        self.register_transition('global_feature_selection', Role.COORDINATOR)
        self.register_transition('validate', Role.PARTICIPANT)

    def run(self):
        # defining the client
        cohort_name = self.id
        client = Client()
        client.config_based_init(
            client_name = cohort_name, 
            input_folder = INPUT_FOLDER
        )

        self.store(key="smpc", value=client.smpc)
        self.store(key='client', value=client)
        self.store(key="separator", value=client.data_separator)
        self.configure_smpc(exponent=13)

        # exchange the batch_labels and covariates
        self.send_data_to_coordinator(
            (client.batch_labels, client.variables),
            send_to_self=True,
            use_smpc=False
        )

        # -------- coordinator --------        
        if self.is_coordinator:
            # union the batch_labels and intersect the covariates
            list_labels_variables = self.gather_data(is_json=False)
            first_labels, first_variables = list_labels_variables[0]
            global_variables = set(first_variables)
            global_batch_labels = list(first_labels)
            
            for labels, variables in list_labels_variables[1:]:
                global_variables.intersection_update(variables)
                global_batch_labels.extend(labels)
                
            # Check for uniqueness in batch labels
            if len(global_batch_labels) != len(set(global_batch_labels)):
                raise ValueError("Batch labels are not unique across clients, please adjust them")
            # send around the number of batches and the variables            
            self.broadcast_data((global_variables, len(global_batch_labels)),
                                send_to_self=True,
                                memo="commonVariables")

        # -------- participant --------
        # we receive the common variables and the number of batches
        global_variables, num_batches = self.await_data(n=1, is_json=False, memo="commonVariables")
        self.store(key='global_variables', value=global_variables)
        # now we can calculate the min_samples_per_feature
        # min_samples = max(num_batches + len(global_variables) + 1, client.min_samples)
        batch_feature_presence = client.get_batch_feature_presence_info(min_samples=client.min_samples)

        self.send_data_to_coordinator((cohort_name,
                                       client.position,
                                       client.reference_batch,
                                       batch_feature_presence),
                                    send_to_self=True,
                                    use_smpc=False)
        if self.is_coordinator:
            return 'global_feature_selection'
        return 'validate'


@app_state('global_feature_selection')
class GlobalFeatureSelection(AppState):
    def register(self):
        self.register_transition('validate', Role.COORDINATOR)

    def run(self):
        # wait for each client to send the list of genes they have
        logging.info("[global_feature_selection] Waiting for clients to send their data")
        feature_information = self.gather_data(is_json=False)
        logging.info("[global_feature_selection] Received data from all clients")

        assert self._app is not None
        global_feature_names, feature_presence_matrix, cohorts_order = \
              c_utils.select_common_features_variables(feature_information,
                                                       default_order=self._app.clients,
                                                       min_clients=len(self._app.clients))
        self.broadcast_data((global_feature_names, cohorts_order),
                            send_to_self=True, memo="common_features")
        self.store(key='feature_presence_matrix', value=feature_presence_matrix)
        logging.info(
            "[global_feature_selection] Transitioning to validation step."
        )
        return 'validate'

@app_state('validate')
class ValidationState(AppState):
    def register(self):
        self.register_transition('first_step', Role.BOTH)

    def run(self):
        # obtain and safe common genes and indices of design matrix
        logging.info("[validate] waiting for common features and covariates")
        global_feature_names, cohorts_order = self.await_data(n=1, is_json=False, memo="common_features")
        global_variables = self.load("global_variables")
        client = self.load('client')

        client.validate_inputs(global_variables)
        logging.info("[validate] Inputs have been validated")
        client.set_data(global_feature_names)
        logging.info("[validate] Data has been set to contain all global features")

        # get all client names to generate design matrix
        client.create_design(cohorts_order)
        logging.info(f"[validate] Design matrix has been created with shape {client.design.shape}")
        logging.info("[validate] design has been created")
        self.store(key='client', value=client)
        return 'first_step'


@app_state('first_step')
class FirstCombatStep(AppState):
    def register(self):
        self.register_transition('get_estimates', Role.PARTICIPANT)
        self.register_transition('compute_b_hat', Role.COORDINATOR)

    def run(self):
        logging.info("[ComBat-first_step:] Starting the first step of ComBat")
        client = self.load('client')
        logging.info(f"[ComBat-first_step:] Adjusting for {len(client.variables)} covariate(s) or covariate level(s)")
        if client.mean_only:
            logging.info("[ComBat-first_step:] Performing ComBat with mean only.")
        
        # getting XtX and Xty
        XtX, XtY = client.compute_XtX_XtY()
        design_cols = client.design.shape[1]
        ref_size = [sum(client.design.iloc[:, i]) for i in range(design_cols - len(client.variables))]

        # save XtX and XtY
        self.store(key="XtX", value=XtX)
        self.store(key="XtY", value=XtY)

        # send the data to the coordinator
        self.send_data_to_coordinator([XtX, XtY, ref_size], send_to_self=True, use_smpc=client.smpc)
        logging.info("[ComBat-first_step:] Computation done, sending data to coordinator")
        logging.info(f"[ComBat-first_step:] XtX of shape {XtX.shape}, X of shape {client.design.shape}, XtY of shape {XtY.shape}")

        # If the client is the coordinator, we can move to the next step
        if self.is_coordinator:
            return 'compute_b_hat'
        return 'get_estimates'


@app_state('compute_b_hat')
class ComputeBHatState(AppState):
    def register(self):
        self.register_transition('get_estimates', Role.COORDINATOR)

    def run(self):
        logging.info("[Compute_b_hat] Computing b_hat")
        client = self.load('client')
        n = client.data.values.shape[0]
        k = client.design.shape[1]
        XtX_XtY_lists = self.gather_data(is_json=False, use_smpc=client.smpc)
        self.log("[ComBat-first_step:] Got XtX_XtY_list from gathered data.")

        # Aggregate the XtX and XtY values from all clients
        XtX_global, XtY_global, ref_size = c_utils.aggregate_XtX_XtY(XtX_XtY_lists, n=n, k=k, use_smpc=client.smpc)
        self.store(key="XtX_global", value=XtX_global)
        self.store(key="XtY_global", value=XtY_global)
        self.log("[Compute_b_hat:] XtX_global and XtY_global have been aggregated.")
        self.log(f"[Compute_b_hat:] Ref size: {ref_size}")

        # Compute B.hat = inv(ls1) @ ls2.
        B_hat = c_utils.compute_B_hat(XtX_global, XtY_global)
        self.store(key="B_hat", value=B_hat)
        self.log("[Compute_b_hat:] B_hat has been computed.")
        # Compute the grand mean and stand_mean
        grand_mean, stand_mean = c_utils.compute_mean(XtX_global, XtY_global, B_hat, ref_size)
        self.store(key="grand_mean", value=grand_mean)
        self.store(key="stand_mean", value=stand_mean)
        self.log("[Compute_b_hat:] Grand mean and stand mean have been computed.")

        # send B_hat and stand_mean to all clients
        self.broadcast_data([B_hat, stand_mean, ref_size], send_to_self=True, memo="B_hat")
        return 'get_estimates'


@app_state('get_estimates')
class GetEstimatesState(AppState):
    def register(self):
        self.register_transition('standardize_data', Role.PARTICIPANT)
        self.register_transition('pooled_variance', Role.COORDINATOR)

    def run(self):
        logging.info("[get_estimates] Getting L/S estimates...")
        client = self.load('client')        
        # get the B_hat and stand_mean
        B_hat, stand_mean, ref_size = self.await_data(n=1, is_json=False, memo="B_hat")

        # save the B_hat and stand_mean
        self.store(key="B_hat", value=B_hat)
        self.store(key="stand_mean", value=stand_mean)
        self.store(key="ref_size", value=ref_size)

        # get the L/S estimates
        sigma_site = client.get_sigma_summary(B_hat, ref_size)
        self.store(key="sigma_site", value=sigma_site)
        self.log(f"[get_estimates:] L/S estimates have been computed. Sigma site has been computed, shape: {sigma_site.shape}")

        # send the sigma_site to the coordinator
        self.send_data_to_coordinator(sigma_site  * client.data.shape[1], send_to_self=True, use_smpc=client.smpc)
        self.log("[get_estimates:] Sigma site has been sent to the coordinator.")
        if self.is_coordinator:
            return 'pooled_variance'
        return 'standardize_data'

@app_state('pooled_variance')
class PooledVarianceState(AppState):
    def register(self):
        self.register_transition('standardize_data', Role.COORDINATOR)

    def run(self):
        logging.info("[pooled variance] Getting pooled variance")
        var_list = self.gather_data(is_json=False, use_smpc=self.load("smpc"))

        # get the pooled variance
        pooled_variance = c_utils.get_pooled_variance(var_list, self.load("ref_size"), self.load("smpc"))
        self.store(key="pooled_variance", value=pooled_variance)
        self.log(f"[pooled variance:] Pooled variance has been computed, shape: {pooled_variance.shape}")

        # send the pooled variance to all clients
        self.broadcast_data(pooled_variance, send_to_self=True, memo="pooled_variance")
        return 'standardize_data'

@app_state('standardize_data')
class CalculateEstimatesState(AppState):
    def register(self):
        self.register_transition('apply_correction', Role.BOTH)

    def run(self):
        logging.info("[calculate_estimates] Calculating estimates")
        pooled_variance = self.await_data(n=1, is_json=False, memo="pooled_variance")
        self.log("[calculate_estimates] Pooled variance has been received.")
        self.store(key="pooled_variance", value=pooled_variance)

        client = self.load('client')
        self.log("[calculate_estimates] Getting standardized data...")
        client.get_standardized_data(
            self.load("B_hat"),
            self.load("stand_mean"),
            self.load("pooled_variance"),
            self.load("ref_size")
        )
        self.log("[calculate_estimates] Standardized data has been computed.")

        # Get naive estimators
        client.get_naive_estimates()

        if client.eb_param:
            if client.parametric:
                self.log("[calculate_estimates] Getting parametric Empirical Bayes estimates...")
            else:
                self.log("[calculate_estimates] Getting non-parametric Empirical Bayes estimates...")
            client.get_eb_estimators()
            self.log("[calculate_estimates] Empirical Bayes estimates have been computed.")
        else:
            client.gamma_star = client.gamma_hat.copy()
            client.delta_star = client.delta_hat.copy()
            self.log("[calculate_estimates] Non-Empirical Bayes estimates have been computed.")

        return 'apply_correction'

@app_state('apply_correction')
class ApplyCorrectionState(AppState):
    def register(self):
        self.register_transition('terminal')

    def run(self):
        logging.info("[apply_correction] Applying batch correction")
        client = self.load('client')

        pooled_variance = self.load("pooled_variance")
        # remove the batch effects in own data and safe the results
        corrected_data = client.get_corrected_data(pooled_variance)

        logging.info("[apply_correction] Batch correction has been applied.")
        corrected_data.to_csv(
            os.path.join(os.getcwd(), "mnt", "output", "batch_corrected_data.csv"),
            sep=self.load("separator")
    )
        # with open(os.path.join(os.getcwd(), "mnt", "output", "report.txt"), "w") as f:
        #     f.write(client.report)
        return 'terminal'