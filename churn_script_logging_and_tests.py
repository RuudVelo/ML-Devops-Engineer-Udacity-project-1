"""
This file contains the unit tests and loggings from a churn_library.py script

Author: Ruud Goorden

Date: 27-3-2023
"""

# import libraries
import os
import logging
import churn_library as cls

def test_import(import_data):
    """
    test data import function
    """
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return df


def test_eda(perform_eda, df):
    """
    test perform_eda function
    """
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    perform_eda(df)
    path = "./images/eda"

    # Checking if the list is empty or not
    try:
        # Getting the list of directories
        dir_val = os.listdir(path)
        assert len(dir_val) > 0
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.warning("Testing perform_eda: Images seem not to be saved correctly "
                        "in the eda folder.")
        raise err


def test_encoder_helper(_encoder_helper, df):
    """
    test encoder_helper function
    """
    cat_columns = ['Card_Category', 'Education_Level', 'Gender', 'Income_Category', 
                   'Marital_Status',
                   ]

    df = _encoder_helper(df, cat_columns, 'Churn')

    try:
        for col in cat_columns:
            assert col in df.columns
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: Transformed categorical columns "
            "are missing in the dataframe")
        return err

    return df


def test_perform_feature_engineering(perform_feature_engineering, df):
    """
    test perform_feature_engineering function
    """
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, 'Churn')
    try:
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: "
                      "The 4 objects are not returned")
        raise err

    return X_train, X_test, y_train, y_test


def test_train_models(train_models, X_train, X_test, y_train, y_test):
    """
    test train_models function
    """
    train_models(X_train, X_test, y_train, y_test)
    path = "./images/results/"
    try:
        # Get the list of directories
        dir_val = os.listdir(path)
        assert len(dir_val) > 0
    except FileNotFoundError as err:
        logging.error("Testing train_models: Cannot find the result images")
        raise err

    path = "./models/"
    try:
        # Getting the list of directories
        dir_val = os.listdir(path)
        assert len(dir_val) > 0
        logging.info("Testing train_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models: Cannot find model files")
        raise err


if __name__ == "__main__":
    DATA_FRAME = test_import(cls.import_data)
    test_eda(cls.perform_eda, DATA_FRAME)
    DATA_FRAME = test_encoder_helper(cls._encoder_helper, DATA_FRAME)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(
        cls.perform_feature_engineering, DATA_FRAME)
    test_train_models(cls.train_models, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
