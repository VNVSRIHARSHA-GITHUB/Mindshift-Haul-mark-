# Mindshift Analytics Haul Mark Challenge 

This repository contains the machine learning pipelines and analytical reporting scripts for the **Mindshift Analytics Haul Mark Challenge**. 

The solution utilizes an ensemble of gradient boosting models (CatBoost, LightGBM, and XGBoost) optimized with Optuna to predict fuel consumption (`acons`). Additionally, it generates secondary analytical reports to assess dumper and operator efficiency.

##  Repository Contents

* **`mainoutput.py`**: The core predictive pipeline. It loads the summary data and cached features, optimizes a CatBoost model using Optuna, trains an ensemble of LightGBM, XGBoost, and CatBoost models using 5-Fold Cross-Validation, and generates the final Kaggle submission file.
* **`secondaryoutputs.py`**: An extended version of the pipeline that calculates benchmarking metrics. It generates secondary CSV reports detailing efficiency residuals for different dumpers and operators, as well as daily consistency metrics.

---

##  How to Implement in a Kaggle Notebook

Since these scripts use hardcoded Kaggle directories (`/kaggle/input/...` and `/kaggle/working/...`), they are designed to be run directly inside a Kaggle Notebook.

### Step 1: Set Up Your Kaggle Environment
1. Log into [Kaggle](https://www.kaggle.com/) and create a **New Notebook**.
2. Set the environment to **Python**.
3. (Optional but recommended) Turn on **GPU** or **TPU** in the notebook settings (Notebook Options -> Accelerator) to speed up model training, especially for CatBoost and XGBoost.

### Step 2: Attach the Required Datasets
Your scripts rely on specific input paths. You must attach the following data sources to your notebook:

1.  **Competition Data:** * Click **"Add Input"** -> Search for the competition `mindshift-analytics-haul-mark-challenge` and attach it.
    * *Path expected by script:* `/kaggle/input/competitions/mindshift-analytics-haul-mark-challenge`
2.  **Cached Features Dataset:**
    * Click **"Add Input"** -> Search for the dataset `nvdpda24b046` (which contains `cached_features.parquet`) and attach it.
    * *Path expected by script:* `/kaggle/input/datasets/nvdpda24b046/cached-features-parquet/cached_features.parquet`
3.  **ID Mapping:**
    * Ensure `id_mapping_new.csv` is present in the competition directory. If it is a custom file, upload it as a new dataset and update the `ID_MAPPING_PATH` variable in the scripts to point to your new dataset.

### Step 3: Install Dependencies
Kaggle environments come with most libraries pre-installed, but it is highly recommended to ensure you have the latest versions of the gradient boosting libraries and Optuna. Add a code cell at the top of your notebook and run:

```python
!pip install lightgbm xgboost catboost optuna -q
