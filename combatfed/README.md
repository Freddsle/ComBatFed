<table>
  <tr>
    <td><a href="https://freddsle.github.io/ComBatFed/combatfed/#usage"><img src="https://img.shields.io/badge/HowTo_Guide-Click_Here!-007EC6?style=for-the-badge" alt="HowTo Guide"></a></td>
    <td><a href="https://freddsle.github.io/ComBatFed/"><img src="https://img.shields.io/badge/Documentation-Click_Here!-007EC6?style=for-the-badge" alt="Documentation"></a></td>
    <td><a href="https://github.com/Freddsle/ComBatFed/"><img src="https://img.shields.io/badge/GitHub-Click_Here!-007EC6?style=for-the-badge" alt="GitHub"></a></td>
    <td><a href="https://featurecloud.ai/app/combatfed"><img src="https://img.shields.io/badge/FeatureCloud_App-Click_Here!-007EC6?style=for-the-badge" alt="FeatureCloud App"></a></td>
  </tr>
</table>


# Federated batch effects correction with ComBat (ComBatFed) <!-- omit in toc -->

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Coverage Status](https://coveralls.io/repos/github/Freddsle/ComBatFed/badge.svg?branch=main)](https://coveralls.io/github/Freddsle/ComBatFed?branch=main)

---

## Table of Contents  <!-- omit in toc -->
- [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Clone the repository](#clone-the-repository)
- [File Structure](#file-structure)
    - [Input](#input)
    - [Configuration](#configuration)
- [Usage](#usage)
    - [Quick start - Test Mode](#quick-start---test-mode)
    - [Quick start - Collaboration Mode](#quick-start---collaboration-mode)
- [FeatureCloud App states](#featurecloud-app-states)
- [Contact information](#contact-information)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [How to cite](#how-to-cite)

---

The **ComBatFed** is a federated implementation of the 'ComBat' method from the 'sva' R package, developed within the [FeatureCloud](https://featurecloud.ai/) platform. It enables privacy-preserving batch effect correction by keeping raw data decentralized and utilizing Secure Multiparty Computation (SMPC) for secure data aggregation.

ComBatFed allows multiple participants to collaboratively remove batch effects from their data without sharing raw data, ensuring privacy.  
You can access and use the `ComBatFed` app directly on [FeatureCloud](https://featurecloud.ai/app/combatfed). 

---

## Installation
 
### Prerequisites
 
 Before installing `ComBatFed`, ensure you have the following installed:
 1. **Docker**: [Installation Instructions](https://www.docker.com/get-started).
 2. **FeatureCloud CLI**.
    - Install FeatureCloud CLI using pip:
      ```bash
      pip install featurecloud
      ```
    - Run controller:
      ```bash
      featurecloud controller start
      ```
    For Windows users, git must also be installed and added to PATH. We recommend
    and tested using [WSL](https://docs.docker.com/desktop/features/wsl/).

 3. **App Image**:
   - For linux/amd64:
     ```bash
     # pull the pre-built image
     featurecloud app download featurecloud.ai/combatfed
     ```
     or directly via Docker
     ```bash
     docker pull featurecloud.ai/combatfed:latest
     ```
   - Alternatively, If you are using a ARM architecture (e.g., Mac M-series), you may need to build the image locally as shown below.  
     ```bash
     docker build . -t featurecloud.ai/combatfed:latest
     ```

     or build the image from GitHub locally:
     ```bash
      git clone https://github.com/Freddsle/ComBatFed.git
      cd ComBatFed/combatfed
      docker build . -t featurecloud.ai/combatfed:latest
      ```

 The app image which is provided in the docker registry of featurecloud built on the linux/amd64 platform. Especially if you're using a Macbook with any of the M-series chips or any other device not compatible with linux/amd64, please build the image locally.

### Clone the repository
 
 If you want to run the simulations locally, clone the repository:
 
 ```bash
 git clone https://github.com/Freddsle/ComBatFed.git
 cd ComBatFed
 ```
 
 This will clone the repository to your local machine with example files and simulation scripts.
 
---
 
## File Structure
### Input 

 In summary, you need two main inputs and one optional file:

<p align="center">
   <img src="https://github.com/Freddsle/ComBatFed/blob/main/docs/fig_S1.jpg?raw=true" alt="Required files figure" width="90%">
   <br>
   <em>Input files required for ComBatFed.</em>
</p>


- **Data File**: The main data file containing the expression data. It should be in a tabular format (e.g., TSV, CSV) with samples as rows and features as columns or vice versa. The first column should contain the sample names / feature names (depending on the `rows_as_features` parameter in the config file).  
The data file can contain missing values, but they will be removed on the first step of the app.
- **Design File**: An optional file that specifies the design matrix for the data. It should contain sample names and covariates.
The first column should contain the sample names, that match the sample names in the data file. The design file is read in the following way:
- **Configuration File**: A YAML file that specifies the parameters for the ComBatFed app. This file should be named `config.yml` or `config.yaml` and placed in the input folder. 

**Minimal Example Directory Structure**:
```text
client_folder/
├─ config.yml
├─ expression_data.csv
├─ design.csv
```
 
### Configuration
 
 `ComBatFed` is highly configurable via the `config.yml` file. This file controls data formats, normalization methods, and other essential parameters.
 
 Example Config File (config.yml):
 
 ```yaml

  FedComBat:                                    # Mandatory header
  # Required settings:
  data_filename: "expr_for_correction.tsv"      # Data file relative to the input folder
  data_separator: "\t"                          # CSV file delimiter

  # Optional settings:
  min_samples: 3                                # Minimum samples required per feature
  covariates: ["Status"]                        # List of covariates to use
  smpc: true                                    # Flag for secure multi-party computation
  design_filename: "design.tsv"                 # Design file (optional; required if using batch info)
  design_separator: "\t"                        # Delimiter for the design file
  rows_as_features: false                       # Set to true if the data file is an expression file with features as rows
  index_col: 0                                  # Column to use as index (0-based)
  position: 1                                   # Client position (if applicable)
  batch_col_name: "batch"                       # Column in the design file that contains batch information

  output_tabular: true                          # Output tabular data, if true: sample x features; if false: features x sample

 ```

---

## Usage

To run the ComBatFed app, you can use the FeatureCloud CLI or the FeatureCloud web interface. 

The app is designed to be run in a Docker container, which ensures that all dependencies are included and that the environment is consistent across different machines. So, it is mandatory to have Docker installed and running.

The app can be run in a test environment or in a collaboration mode:
- **Test Mode**: You can run ComBatFed as a standalone app in the [FeatureCloud test-bed](https://featurecloud.ai/development/test). This mode is used for testing and simulating the app's functionality. It allows you to run the app locally without needing to set up a full collaboration environment or multiple machines. **No registration is needed.**
- **Collaboration Mode**: You can run ComBatFed as as a [FeatureCloud Workflow](https://featurecloud.ai/projects). This mode is used for real-world applications where multiple participants collaborate to correct batch effects in their data. It requires multiple machines with running FeatureCloud.

--> Go to the [Test Mode](#quick-start-test-mode) section for a quick start guide.  
--> Go to the [Collaboration Mode](#quick-start-collaboration-mode) section for a quick start guide.

For any scenario, make sure that [Pre-requisites](#prerequisites) are met and that the input files are correctly formatted and placed in the input folder/

### Quick start - Test Mode

 **No registration on FeatureCloud is needed.**

 To run the ComBatFed app in a test environment, follow these steps:
 
 1. **Ensure the full repository including sample data is cloned and the current working directory**: 
 
   ```bash
   git clone https://github.com/Freddsle/ComBatFed.git
   cd ComBatFed
   ```

 2. **Start the FeatureCloud Controller with the correct input folder**:

   ```bash
   # if you have the controller running in a different folder, stop it first
   # featurecloud controller stop
   featurecloud controller start --data-dir=./datasets/Ecoli_proteomics/before
   ```

   This command starts the FeatureCloud controller and sets the data directory to the specified path. The `--data-dir` option specifies the directory where the input files are located. The `--data-dir` should point to the folder containing the input files, including `config.yml`, data files, and design files.

   If the controller is running, you will see the yellow / green icon on the FeatureCloud web interface:

   <p align="center">
      <img src="https://github.com/Freddsle/ComBatFed/blob/main/docs/controller.png?raw=true" alt="Controller." width="50%">
      <br>
      <em>GUI for the running controller.</em>
   </p>


   If the controller is not running, you will see a red icon. 


 3. **Run a Sample Experiment**:  

   ```bash
   featurecloud test start --app-image=featurecloud.ai/combatfed:latest --client-dirs=lab_A,lab_B,lab_C,lab_D,lab_E
   ```

   Alternatively, you can start the experiment from the [frontend](https://featurecloud.ai/development/test/new):
   - Use `featurecloud.ai/combatfed:latest` as the app image.
   - Select 5 clients, add lab_A, lab_B, lab_C, lab_D, lab_E respectively for the 5 clients to their path. 
   - Click "Start" to run the app.

   <p align="center">
      <img src="https://github.com/Freddsle/ComBatFed/blob/main/docs/start_test1.png?raw=true" alt="Test GUI." width="50%">
      <br>
      <em>Example test.</em>
   </p>

4. **Monitor the Experiment**:
    - You can monitor the progress of the experiment in the FeatureCloud web interface. The app will run on each client, and you can view logs and results as they are generated.
    - The app will run for a few minutes, depending on the size of the data and the number of clients. You can check the logs for any errors or warnings.
    
 
### Output

Once the app has finished running, you will see a summary of the results in the web interface.

<p align="center">
   <img src="https://github.com/Freddsle/ComBatFed/blob/main/docs/test_running.png?raw=true" alt="Test GUI" width="60%">
   <br>
   <em>Where to find results and logs.</em>
</p>


Output files include:
- **Corrected Data**: The batch-corrected data, provided in the same format as the input file or as specified in the configuration file.
- **Log File**: A detailed log of the processing steps and any warnings or errors encountered.
 
Alternatively, you can check the logs in the terminal where you started the FeatureCloud controller (logs folder), and find the results in the test folder there.


## Quick start - Collaboration Mode
 
**Registration on FeatureCloud is needed.**

For an actual multi-party setting:
1. **Coordinator creates a Project** in [FeatureCloud](https://featurecloud.ai/projects) and invite at least 3 clients (distributes tokens).
2. **Each Participant (Client)** prepares their data and a `config.yml` file.
3. **All Clients Join with Tokens** provided by the coordinator.
4. **Each Client** uploads their data and `config.yml` to their local FeatureCloud instance.
5. **Start the Project**: `ComBatFed` runs securely, never sharing raw data.
6. **Get the Results**: Each client receives the corrected data and logs.


### Step-by-step scenario

**Scenario**: Three clients (A, B, and C) collaborate on a federated analysis. Video tutorial: [link](https://featurecloud.ai/researchers).

1. **Coordinator Actions**:  
   - The coordinator logs into the FeatureCloud platform and **creates a new project**.
   - Add the ComBatFed app into the workflow and *finalize the project*.
   - The coordinator **creates tokens** and sends them to Clients A, B, and C.

   <p align="center">
      <img src="https://github.com/Freddsle/ComBatFed/blob/main/docs/how_to1.png?raw=true" alt="Coordinator steps." width="70%">
      <br>
      <em>Coordinator steps.</em>
   </p>
   
2. **Client Setup**:
   - Adjust `config.yml` parameters as needed.
   - **Client A, B, C**: Place `expression_data_client.csv` and `config.yml` in a local folder.
   - In case of multiple batches in one client, the client should provide a `design.csv` file with batch information and specify this column name in the `config.yml` parameter `batch_col`.
   
3. **Joining the Project**:
   - Each client uses the FeatureCloud to login and join the project using the provided token.
   - After joining, each client uploads their data and config file to the FeatureCloud GUI client. The data will not be sent to the coordinator or other clients, but makes it available for the local Docker container with the app.

   <p align="center">
      <img src="https://github.com/Freddsle/ComBatFed/blob/main/docs/how_to2.png?raw=true" alt="Client steps." width="70%">
      <br>
      <em>Client steps.</em>
   </p>

4. **Running ComBatFed**:
   - After all clients join, the coordinator starts the project.
   - The app runs locally at each client, securely combining results.
   
### Results and output

After completion, each client finds:
   - The batch-corrected expression data.
   - logs: Detailed logs of the process.
   

## FeatureCloud App states

The app has the following states:

<p align="center">
   <img src="https://github.com/Freddsle/ComBatFed/blob/main/combatfed/myplot.png?raw=true" alt="ComBatFed app states." width="50%">
   <br>
   <em>ComBatFed app states.</em>
</p>

## Contact information
 
 For questions, issues, or support, please open an issue on the [GitHub repository](https://github.com/Freddsle/ComBatFed).
 
 ---
 
## Troubleshooting
 
 Encountering issues? Here are some common problems and their solutions:
 
 - **Missing Files**: Ensure `config.yml` and data files are in the correct directory.
 - **Incorrect Format**: Verify `rows_as_features` and `index_col` settings in `config.yml`.
 - **No Output Produced**: Check logs for error messages.
 - **Errors with Test runs**: Ensure the is no leftover running Docker containers. Restart Docker / System if necessary. 
 
 ---
 
## License
 
 This project is licensed under the [Apache License 2.0](LICENSE).
 
 ---
 
## How to cite
 
 If you use this code in your research, please cite the repository
 ```bibtex
 @misc{ComBatFed,
   author = {Yuliya Burankova},
   title = {ComBatFed: Federated batch effects correction with ComBat},
   year = {2025},
   publisher = {GitHub},
   journal = {GitHub repository},
   url = {https://github.com/Freddsle/ComBatFed}
 }
 ```







