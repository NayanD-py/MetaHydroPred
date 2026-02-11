# Date: 3 Aug 2024
# Author: Nayan Dash
# Title: Script for predicting current density using MetaHydroPred: a Meta-learning framework for predicting current density.
#------------------------------------------------------------------------------

# IMPORT LIBRARIES
import os
import gc
import json
import argparse
import scipy
import pandas as pd
import numpy as np
import joblib
import sklearn
from collections import defaultdict

# import seaborn as sns
# import matplotlib.pyplot as plt

import xgboost
import catboost
import lightgbm
import glob
import pprint

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import LeaveOneOut, KFold, StratifiedKFold

# IMPORT MODElS
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,  HistGradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import clone

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer, PowerTransformer, QuantileTransformer, MaxAbsScaler

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PolynomialFeatures


from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score, explained_variance_score,
                             max_error, median_absolute_error, mean_squared_log_error)
from scipy.stats import pearsonr

#-----------------------------------------------------------------------------------------------
import warnings

warnings.filterwarnings('ignore')

print("#"*50)
print("scikit-learn version:", sklearn.__version__)
print("xgboost version:", xgboost.__version__)
print("catboost version:", catboost.__version__)
print("lightgbm version:", lightgbm.__version__)
print("numpy version:", np.__version__)
print("scipy version:", scipy.__version__)
print("joblib version:", joblib.__version__)

print("#"*50)


###########################################################################


# GENERATE META-FEATURES USING LEAVE-ONE-OUT CROSS-VALIDATION (LOOCV)
def generate_meta_features(X_train, y_train, X_test, y_test, saved_baseline_models_dir):

    # Get the list of models
    model_list = glob.glob(saved_baseline_models_dir)

    # Create a dictionary using the first two words of the file name as keys and load the models
    base_models = []

    for model_path in model_list:

        model_name = '_'.join(os.path.basename(model_path).replace('.joblib', '').split('_')[:2])
        model = joblib.load(model_path)
        base_models.append((model_name, model))

    # Convert to arrays
    X_train_values = X_train
    y_train_values = y_train.values
    X_test_values = X_test


    # Initialize arrays to store meta-features
    meta_features_train = np.zeros((X_train_values.shape[0], len(base_models)))
    meta_features_test = np.zeros((X_test_values.shape[0], len(base_models)))
    
    loo = LeaveOneOut()
    
    for i, (name, model) in enumerate(base_models):

        model = clone(model)

        print(model)

        oof_predictions = np.zeros(X_train.shape[0])
        test_predictions = np.zeros((X_test.shape[0], X_train.shape[0]))  # LOOCV has as many splits as there are samples
        
        for j, (train_idx, valid_idx) in enumerate(loo.split(X_train)):
            X_train_fold, X_valid_fold = X_train_values[train_idx], X_train_values[valid_idx]
            y_train_fold, y_valid_fold = y_train_values[train_idx], y_train_values[valid_idx]
            
            print(f"Fitting model: {name}")
            
            # Fit the model on the current fold
            model.fit(X_train_fold, y_train_fold)
            
            # Generate out-of-fold predictions for the training set
            oof_predictions[valid_idx] = model.predict(X_valid_fold)
            
            # Generate predictions for the test set
            test_predictions[:, j] = model.predict(X_test_values)
        
        # Store the out-of-fold predictions as meta-features for training
        meta_features_train[:, i] = oof_predictions
        
        # Store the average of the test set predictions as meta-features for testing 
        meta_features_test[:, i] = test_predictions.mean(axis=1)
    
    # Convert to DataFrame with appropriate column names
    meta_features_train_df = pd.DataFrame(meta_features_train, columns=[name for name, _ in base_models])
    meta_features_test_df = pd.DataFrame(meta_features_test, columns=[name for name, _ in base_models])
    
    # Combine the original features with the meta-features
    X_train_combined = pd.concat([meta_features_train_df, y_train], axis=1)
    X_test_combined = pd.concat([meta_features_test_df, y_test], axis=1)
    
    return X_train_combined, X_test_combined


#------------------------------------------------------------------------------
def meta_dataset_preparation(train_dataset_path, test_dataset_path, baseline_models_dir, drop_features=None, target_column='current_density'):


    # Load the training dataset
    if train_dataset_path is not None:
        train_data = pd.read_csv(train_dataset_path, header=0)
    else:
        raise ValueError("Training dataset path must be provided.")
    # Load the test dataset
    if test_dataset_path is not None:
        test_data = pd.read_csv(test_dataset_path, header=0)
    else:
        raise ValueError("Test dataset path must be provided.")
    

    # Check if selected features are provided
    if drop_features is not None:
        # Filter the datasets to include only the selected features
        train_data = train_data.drop(columns=drop_features, errors='ignore')
        test_data = test_data.drop(columns=drop_features, errors='ignore')


    # Extract the target variable from the training and test datasets
    train_X = train_data.drop(columns=[target_column])
    train_y = train_data[target_column]

    test_X = test_data.drop(columns=[target_column])
    test_y = test_data[target_column]


    scaler = MinMaxScaler()
    train_X_norm = scaler.fit_transform(train_X)
    test_X_norm = scaler.transform(test_X)

    train_y_norm = train_y.apply(np.log1p)
    test_y_norm = test_y.apply(np.log1p)

    # Get train and test meta features using LOOCV
    meta_train, meta_test = generate_meta_features(train_X_norm, train_y_norm, test_X_norm, test_y_norm, baseline_models_dir)

    return meta_train, meta_test

#--------------------------------------------------------------------------------
def predict_current_density_all_organic(
    train_dataset_path,
    test_dataset_path,
    meta_train_path,
    target_column,
    baseline_models_dir_BF1,
    baseline_models_dir_BF2,
    meta_model_path,
    output_csv_path,
):
    # Generate meta-features from BF-1
    _ , MF1_test = meta_dataset_preparation(
        train_dataset_path, test_dataset_path,
        baseline_models_dir=baseline_models_dir_BF1,
        drop_features=None,
        target_column=target_column
    )

    # Generate meta-features from BF-2
    _ , MF2_test = meta_dataset_preparation(
        train_dataset_path, test_dataset_path,
        baseline_models_dir=baseline_models_dir_BF2,
        drop_features=["Substrate concentration", "Reactor working volume"],
        target_column=target_column
    )

    # Rename columns
    MF1_test = MF1_test.rename(columns={col: f"{col}_stk" for col in MF1_test.columns if col != target_column})
    MF2_test = MF2_test.drop(columns=target_column, axis=1).rename(columns=lambda x: f"{x}_corr_stk")

    # Combine MF1 and MF2
    MF5_test = pd.concat([MF2_test, MF1_test], axis=1)

    # Load the training meta-features
    MF5_train = pd.read_csv(meta_train_path, header=0)

    # Reorder the columns to match the test set
    MF5_test = MF5_test.reindex(columns=MF5_train.columns)

    # Prepare training and testing data
    MF5_test_X = MF5_test.drop(columns=target_column, axis=1)
    MF5_train_X = MF5_train.drop(columns=target_column, axis=1)
    MF5_train_y = MF5_train[target_column]

    # Scale the features
    scaler = MinMaxScaler()
    MF5_train_X_norm = scaler.fit_transform(MF5_train_X)
    MF5_test_X_norm = scaler.transform(MF5_test_X)  


    # Load meta model
    meta_model = joblib.load(meta_model_path)
    print("Meta model loaded:", meta_model)

    # Predict
    predictions = meta_model.predict(MF5_test_X_norm)
    predictions = np.expm1(predictions)  # Inverse of log1p

    # Save predictions
    output_df = pd.DataFrame({'Current density': predictions})
    output_df.to_csv(output_csv_path, index=False)

    return predictions  # Optional: return for downstream use

#--------------------------------------------------------------------------------
def predict_current_density_acetate(
    train_dataset_path,
    test_dataset_path,
    meta_train_path,
    target_column,
    baseline_models_dir_BF1,
    baseline_models_dir_BF2,
    meta_model_path,
    output_csv_path,
):
    # Step 1: Generate meta-features from BF-1 and BF-2
    _, MF1_test = meta_dataset_preparation(
        train_dataset_path, test_dataset_path,
        baseline_models_dir=baseline_models_dir_BF1,
        drop_features=None,
        target_column=target_column
    )

    _, MF2_test = meta_dataset_preparation(
        train_dataset_path, test_dataset_path,
        baseline_models_dir=baseline_models_dir_BF2,
        drop_features=["S/V ratio", "Temperature"],
        target_column=target_column
    )



    # Step 2: Rename and drop appropriately
    MF1_test = MF1_test.rename(columns={col: f"{col}_stk" for col in MF1_test.columns if col != target_column})
    MF2_test = MF2_test.drop(columns=target_column).rename(columns=lambda x: f"{x}_corr_stk")

    MF5_test = pd.concat([MF2_test, MF1_test], axis=1)



    # Load the training meta-features
    MF5_train = pd.read_csv(meta_train_path, header=0)

    # Reorder the columns to match the test set
    MF5_test = MF5_test.reindex(columns=MF5_train.columns)

    # Prepare training and testing data
    MF5_test_X = MF5_test.drop(columns=target_column, axis=1)
    MF5_train_X = MF5_train.drop(columns=target_column, axis=1)
    MF5_train_y = MF5_train[target_column]

    # Scale the features
    scaler = MinMaxScaler()
    MF5_train_X_norm = scaler.fit_transform(MF5_train_X)
    MF5_test_X_norm = scaler.transform(MF5_test_X)  


    # Load meta model
    meta_model = joblib.load(meta_model_path)
    print("Meta model loaded:", meta_model)


    predictions = meta_model.predict(MF5_test_X_norm)
    predictions = np.expm1(predictions)  # Inverse of log1p

    # Step 5: Save predictions
    output_df = pd.DataFrame({'Current density': predictions})
    output_df.to_csv(output_csv_path, index=False)

    return predictions  # Optional return

# --------------------------------------------------------------------------------

def predict_current_density_complex_substrate(
    train_dataset_path,
    test_dataset_path,
    meta_train_path,
    target_column,
    baseline_models_dir_BF1,
    baseline_models_dir_BF2,
    baseline_models_dir_BF3,
    meta_model_path,
    output_csv_path,
):
    # Generate meta-features for the training and test datasets using baseline models
    _, MF1_test = meta_dataset_preparation(train_dataset_path,
                                                                  test_dataset_path,
                                                                    baseline_models_dir=baseline_models_dir_BF1,
                                                                      drop_features=None, 
                                                                        target_column=target_column)
    
    _, MF2_test = meta_dataset_preparation(train_dataset_path,
                                                                  test_dataset_path,
                                                                    baseline_models_dir=baseline_models_dir_BF2,
                                                                      drop_features=["Applied voltage", "Temperature"], 
                                                                        target_column=target_column)
    

    _, MF3_test = meta_dataset_preparation(train_dataset_path,
                                                                test_dataset_path,
                                                                    baseline_models_dir=baseline_models_dir_BF3,
                                                                        drop_features=["Cathode projected surface area", "S/V ratio"], 
                                                                            target_column=target_column)
    

    


    # Rename all other columns except the excluded one
    MF1_test = MF1_test.rename(columns={col: f"{col}_stk" for col in MF1_test.columns if col not in target_column})

    MF2_test = MF2_test.drop(columns= target_column, axis=1)
    MF3_test = MF3_test.drop(columns= target_column, axis=1)


    # Rename columns in corr_train and fi_train to avoid duplication
    MF2_test = MF2_test.rename(columns=lambda x: f"{x}_corr_stk")
    MF3_test = MF3_test.rename(columns=lambda x: f"{x}_fi_stk")

    # Combine the meta-features from all three baseline models
    MF4_test = pd.concat([MF1_test, MF2_test, MF3_test], axis=1)


    # Load the training meta-features
    MF4_train = pd.read_csv(meta_train_path, header=0)

    # Reorder the columns to match the test set
    MF4_test = MF4_test.reindex(columns=MF4_train.columns)

    # Prepare training and testing data
    MF4_test_X = MF4_test.drop(columns=target_column, axis=1)
    MF4_train_X = MF4_train.drop(columns=target_column, axis=1)
    MF4_train_y = MF4_train[target_column]

    # Scale the features
    scaler = MinMaxScaler()
    MF4_train_X_norm = scaler.fit_transform(MF4_train_X)
    MF4_test_X_norm = scaler.transform(MF4_test_X)  


    # Load meta model
    meta_model = joblib.load(meta_model_path)
    print("Meta model loaded:", meta_model)

    predictions = meta_model.predict(MF4_test_X_norm)
    predictions = np.expm1(predictions)  # Inverse of log1p transformation

    # Save the predictions to a CSV file
    output_df = pd.DataFrame({
        # 'Sample ID': MF5_test.index,
        'Current density': predictions
    })
    output_df.to_csv(output_csv_path, index=False)

    return predictions


#------------------------------------------------------------------------------------


def predict_current_density(
    train_dataset_path,
    test_dataset_path,
    meta_train_path,
    target_column,
    baseline_models_dir_BF1,
    baseline_models_dir_BF2,
    meta_model_path,
    output_csv_path,
    substrate_type,
    baseline_models_dir_BF3=None
    ):
    if substrate_type == "all-organic":
        return predict_current_density_all_organic(
            train_dataset_path=train_dataset_path,
            test_dataset_path=test_dataset_path,
            meta_train_path=meta_train_path,
            target_column=target_column,
            baseline_models_dir_BF1=baseline_models_dir_BF1,
            baseline_models_dir_BF2=baseline_models_dir_BF2,
            meta_model_path=meta_model_path,
            output_csv_path=output_csv_path
        )
    elif substrate_type == "acetate":
        return predict_current_density_acetate(
            train_dataset_path=train_dataset_path,
            test_dataset_path=test_dataset_path,
            meta_train_path=meta_train_path,
            target_column=target_column,
            baseline_models_dir_BF1=baseline_models_dir_BF1,
            baseline_models_dir_BF2=baseline_models_dir_BF2,
            meta_model_path=meta_model_path,
            output_csv_path=output_csv_path
        )
    elif substrate_type == "complex-substrate":
        if baseline_models_dir_BF3 is None:
            raise ValueError("For complex-substrate, baseline_models_dir_BF3 must be provided.")
        else:
            return predict_current_density_complex_substrate(
                train_dataset_path=train_dataset_path,
                test_dataset_path=test_dataset_path,
                meta_train_path=meta_train_path,
                target_column=target_column,
                baseline_models_dir_BF1=baseline_models_dir_BF1,
                baseline_models_dir_BF2=baseline_models_dir_BF2,
                baseline_models_dir_BF3=baseline_models_dir_BF3,  # Optional for complex-substrate
                meta_model_path=meta_model_path,
                output_csv_path=output_csv_path
            )
    else:
        raise ValueError("Invalid substrate type. Choose from 'all-organic', 'acetate', or 'complex-substrate'.")





if __name__ == "__main__":
    import sys
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Predict current density using MetaHydroPred.")
    parser.add_argument('--substrate_type', type=str, choices=['all-organic', 'acetate', 'complex-substrate'], required=True,
                        help="Type of substrate for which to predict current density. Options: 'all-organic', 'acetate', 'complex-substrate'.")
    parser.add_argument('--input_csv_path', type=str, default="../1_data/current-density/benchmark-dataset/current_density_test.csv",
                        help="Path to the input CSV file for predicting.")
    parser.add_argument('--output_csv_path', type=str, default='current_density_predictions.csv',
                        help="Path to save the output CSV file with predictions.")
    # Parse the arguments
    args = parser.parse_args()
    substrate_type = args.substrate_type
    csv_input_path = args.input_csv_path
    target_column = 'Current density'  # Default target column
    output_csv_path = args.output_csv_path

    print(f"Predicting current density for substrate type: {substrate_type}")   



    #--------------------------------------------------------------------------------
    if substrate_type == "all-organic":

        # All-organic 
        train_dataset_path = "../1_data/current-density/benchmark-dataset/current_density_train.csv"
        meta_train_path = "../1_data/current-density/meta-feature-train/all-organic/MF-5-train.csv"

        baseline_models_dir_BF1 = "../3_saved-models/current-density-final/all-organic/baseline-models/BF-1/*.joblib"
        baseline_models_dir_BF2 = "../3_saved-models/current-density-final/all-organic/baseline-models/BF-2/*.joblib"

        meta_model_path = "../3_saved-models/current-density-final/all-organic/meta-model/MetaHydroPred-current-density-all-organic.joblib"

        prediction=predict_current_density_all_organic(
                train_dataset_path=train_dataset_path,
                test_dataset_path=output_csv_path,
                meta_train= meta_train_path,
                target_column=target_column,
                baseline_models_dir_BF1=baseline_models_dir_BF1,
                baseline_models_dir_BF2=baseline_models_dir_BF2,
                meta_model_path=meta_model_path,
                output_csv_path=output_csv_path,
            )
        print(prediction)

    #--------------------------------------------------------------------------------
    elif substrate_type == "acetate":
        # Acetate
        train_dataset_path = "../1_data/current-density/benchmark-dataset/cd_acetate_train.csv"
        meta_train_path = "../1_data/current-density/meta-feature-train/acetate/MF-5-train.csv"


        baseline_models_dir_BF1 = "../3_saved-models/current-density-final/acetate/baseline-models/BF-1/*.joblib"
        baseline_models_dir_BF2 = "../3_saved-models/current-density-final/acetate/baseline-models/BF-2/*.joblib"

        meta_model_path = "../3_saved-models/current-density-final/acetate/meta-model/MetaHydroPred-current-density-acetate.joblib"


        predictions = predict_current_density_acetate(
        train_dataset_path=train_dataset_path,
        test_dataset_path=csv_input_path,
        meta_train_path=meta_train_path,
        target_column=target_column,
        baseline_models_dir_BF1=baseline_models_dir_BF1,
        baseline_models_dir_BF2=baseline_models_dir_BF2,
        meta_model_path= meta_model_path,
        output_csv_path=output_csv_path)   
        print(predictions)

    #--------------------------------------------------------------------------------
    elif substrate_type == "complex-substrate":

        #  Complex-substrate
        train_dataset_path = "../1_data/current-density/benchmark-dataset/cd_complex_substance_train.csv"
        meta_train_path = "../1_data/current-density/meta-feature-train/complex-substrate/MF-4-train.csv"



        baseline_models_dir_BF1 = "../3_saved-models/current-density-final/complex-substrate/baseline-models/BF-1/*.joblib"
        baseline_models_dir_BF2 = "../3_saved-models/current-density-final/complex-substrate/baseline-models/BF-2/*.joblib"
        baseline_models_dir_BF3 = "../3_saved-models/current-density-final/complex-substrate/baseline-models/BF-3/*.joblib"

        meta_model_path = "../3_saved-models/current-density-final/complex-substrate/meta-model/MetaHydroPred-current-density-complex-substrate.joblib"

        predictions = predict_current_density_complex_substrate(
        train_dataset_path=train_dataset_path,
        test_dataset_path=csv_input_path,
        meta_train_path=meta_train_path,
        target_column=target_column,
        baseline_models_dir_BF1=baseline_models_dir_BF1,
        baseline_models_dir_BF2=baseline_models_dir_BF2,
        baseline_models_dir_BF3=baseline_models_dir_BF3,
        meta_model_path= meta_model_path,
        output_csv_path=output_csv_path)
        # Print the predictions   
        print(predictions)
    else:
        print("Invalid substrate type. Please choose from 'all-organic', 'acetate', or 'complex-substrate'.")
        sys.exit(1)
    #--------------------------------------------------------------------------------
        If you want to exit the script after running the predictions, uncomment the following lines
    if not os.path.exists(output_csv_path):
        sys.exit(1)
    print("Current density prediction completed successfully.")
    sys.exit(0)
    








