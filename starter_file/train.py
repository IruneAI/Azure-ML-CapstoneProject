from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

def prepare_data(data):
    #dealing with missing values
    #checking missing
    if (data.isnull().sum()!=0):
        x_df = data.to_pandas_dataframe().dropna()
    #looking into feature types and distribution
    print(data.dtypes)
    #all of them are numerical--> not encoding needed.
    #leave target as it is
    return (x_df, data['target']) 

# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

#creating a tabularDataset
ds =TabularDatasetFactory.from_delimited_files('https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv')### YOUR CODE HERE ###
#cleaning data
x, y = clean_data(ds) 

# TODO: Split data into train and test sets.
#spliting data --> 80-20
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

run = Run.get_context()



def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    os.makedirs('outputs', exist_ok=True)
    #saving model
    joblib.dump(model, 'outputs/model.joblib')
    parser = argparse.ArgumentParser()
# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

#creating a tabularDataset
ds =TabularDatasetFactory.from_delimited_files('https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv')### YOUR CODE HERE ###
#cleaning data
x, y = prepare_data(ds) 

# TODO: Split data into train and test sets.
#spliting data --> 80-20
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

run = Run.get_context()



def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    os.makedirs('outputs', exist_ok=True)
    #saving model
    joblib.dump(model, 'outputs/model.joblib')
    parser = argparse.ArgumentParser()

   

if __name__ == '__main__':
    main()
