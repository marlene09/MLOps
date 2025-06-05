# Import libraries

import os
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import argparse
import logging
import joblib

logging.basicConfig(level=logging.INFO)


def main(args):
    """Main function to train and evaluate the logistic regression model."""
    mlflow.sklearn.autolog()
    df = get_csvs_df(args.training_data)
    x_train, x_test, y_train, y_test = split_data(df)

    model = train_model(args.reg_rate, x_train, y_train)
    evaluate_model(model, x_test, y_test)

# First write a function to read csv files from a folder and concatenate them  then write main


def get_csvs_df(path):
    """Load and concatenate all CSV files from the specified directory into a single DataFrame.

    Args:
        path (str): Path to the directory containing CSV files.

    Returns:
        pd.DataFrame: A concatenated DataFrame from all found CSV files.

    Raises:
        RuntimeError: If the path does not exist or no CSV files are found.

    """
    # Check if the provided path exists
    if not os.path.exists(path):
        # Raise error if path doesn't exist
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")

    # Find all CSV files in the specified directory
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        # Raise error if no CSVs found
        raise RuntimeError(f"No CSV files found in provided data path: {path}")

    # Read and concatenate all CSV files into a single DataFrame
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


# Next split the data into X [independent variables] and y [dependent variable], in this case diabetes


# TO DO: add function to split data
def split_data(df):
    """Split the input DataFrame into independent variables (X) and dependent variables(y) for training.

    The function separates the 'Diabetic' column as the dependent variable,
    and drops 'PatientID' and 'Diabetic' from the independent variable set.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.

    Returns:
        tuple: X_train, X_test, y_train, y_test arrays after splitting.

    Raises:
        RuntimeError: If the expected 'Diabetic' column is not found in the DataFrame.
    """
    df = df.astype({col: 'float64' for col in df.select_dtypes('int').columns})

    if 'Diabetic' not in df.columns:
        raise RuntimeError(
            "Expected target column 'Diabetic' not found in data.")
    X = df.drop(['PatientID', 'Diabetic'], axis=1)
    y = df['Diabetic']
    return train_test_split(X, y, test_size=0.3, random_state=0)


def train_model(reg_rate, x_train, y_train):
    """
    Train a logistic regression model with a given regularization rate.

    Args:
        x_train (array-like): Training data features.
        y_train (array-like): Training data labels.
        reg_rate (float, optional): Regularization rate (lambda). Default is 0.01 later in CL argparser.

    Returns:
        LogisticRegression: A trained LogisticRegression model.
    """
    model = LogisticRegression(C=1 / reg_rate, solver="liblinear")
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    """
    Evaluate the trained model and return accuracy.

    Args:
        model (sklearn.base.BaseEstimator): Trained model with predict method.
        x_test (array-like): Test data features.
        y_test (array-like): True labels for test data.

    Returns:
        float: Accuracy score of the model on test data.
    """
    predictions = model.predict(x_test)          # predictions for x
    accuracy = accuracy_score(y_test, predictions)  # check with ground truth
    logging.info("Model accuracy: %.4f", accuracy)

    # print(f"Model accuracy: {accuracy:.4f}")
    return accuracy





def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01)

    # parse args
    args = parser.parse_args()

    # return args
    return args


# # run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 100)
    print("\n\n")
