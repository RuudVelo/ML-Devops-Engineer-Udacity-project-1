"""
Churn project Udacity

This script contains:
* Importing data
* Performs EDA
* Encodes variables
* Performs feature engineering
* Trains and saves models
* Create feature importance plot
* Create classification report

Author: Ruud Goorden

Date: 27-3-2023
"""

# import libraries
from typing import List, Any
import logging
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split

sns.set()

# Make a logger configuration config
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def import_data(pth: str) -> pd.DataFrame:
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to a csv
    output:
            df: pandas dataframe
    """

    try:
        df = pd.read_csv(pth)
        logging.info("SUCCESS: There are %s rows in the dataframe", df.shape[0])
        logging.info("SUCCES: file is read")
        return df
    except FileNotFoundError:
        logging.error("ERROR: Could not find the file")
        return None


def perform_eda(df: pd.DataFrame) -> None:
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # plot the churn distribution
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.title('Churn distibution')
    plt.savefig('./images/eda/churn_distribution.png')
    plt.close()

    # plot the customer age distribution
    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.title('Customer age distibution')
    plt.savefig('./images/eda/customer_age_distribution.png')
    plt.close()

    # plot the marital status distribution
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.title('Marital status distibution')
    plt.savefig('./images/eda/marital_status_distribution.png')
    plt.close()

    # plot the total transaction cost distribution
    plt.figure(figsize=(20, 10))
    sns.distplot(df['Total_Trans_Ct'])
    plt.title('Total transaction distibution')
    plt.savefig('./images/eda/total_transaction_distribution.png')
    plt.close()

    # plot a heatmap of variables
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.title('Heatmap variables')
    plt.savefig('./images/eda/heatmap.png')
    plt.close()


def _encoder_helper(df: pd.DataFrame, category_lst: List[str], response: str) -> pd.DataFrame:
    """
    Helper function to turn each categorical column into a new column in
    proportion of churn for each category

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name

    output:
            df: pandas dataframe with new columns for each categorical 
            feature in proportion to the response variable
    """
    for col in category_lst:
        new_lst = []
        group_obj = df.groupby(col).mean()[response]

        for val in df[col]:
            new_lst.append(group_obj.loc[val])

        new_col_name = col + '_' + response
        df[new_col_name] = new_lst

    return df


def perform_feature_engineering(
        df: pd.DataFrame,
          response: str
          ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform feature engineering on a pandas DataFrame and split into training and testing sets.

    Args:
        df: A pandas DataFrame containing the data to be processed.
        response: A string indicating the name of the response variable.

    Returns:
        A tuple containing four elements:
        - X_train: A pandas DataFrame representing the training set.
        - X_test: A pandas DataFrame representing the testing set.
        - y_train: A pandas Series representing the response variable in the training set.
        - y_test: A pandas Series representing the response variable in the testing set.
    """
    cat_columns = [
        'Card_Category', 
        'Education_Level',
        'Gender',
        'Income_Category',
        'Marital_Status',
    ]

    keep_cols = [
        'Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
        'Income_Category_Churn', 'Card_Category_Churn'
    ]

    # split dataset for response (dependent) variable y and X independent variables
    y = df['Churn']
    X = pd.DataFrame()

    # apply encoder helper
    df = _encoder_helper(df, cat_columns, response)

    X[keep_cols] = df[keep_cols]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def _classification_report_image(y_train: pd.Series,
                                y_test: pd.Series,
                                y_train_preds_lr: pd.Series,
                                y_train_preds_rf: pd.Series,
                                y_test_preds_lr: pd.Series,
                                y_test_preds_rf: pd.Series) -> None:
    """
    produces classification report for training and testing results and stores report as image
    in images folder.
    input:
            y_train: training response values (pandas Series).
            y_test:  test response values (pandas Series).
            y_train_preds_lr: training predictions from logistic regression (pandas Series).
            y_train_preds_rf: training predictions from random forest (pandas Series).
            y_test_preds_lr: test predictions from logistic regression (pandas Series).
            y_test_preds_rf: test predictions from random forest (pandas Series).

    output:
             None
    """
    # Random forest classification result
    plt.figure()
    plt.rc('figure', figsize=(8, 8))
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.5, str('Random forest results: Test set depicted above and Train set below'), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
        'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/rf_results.png')
    plt.close()

    # Logistic regression classification result
    plt.figure()
    plt.rc('figure', figsize=(8, 8))
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.5, str('Logistic regression results: Test set depicted above and Train set below'), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
        'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/lr_results.png')
    plt.close()


def _feature_importance_plot(model: Any, X_data: pd.DataFrame, output_pth: str) -> None:
    """
    Creates and stores the feature importances plot in the given output path.

    Args:
        model: A trained model object containing the attribute feature_importances_.
        X_data: A pandas dataframe containing the X values.
        output_pth: A string containing the output path to store the plot.

    Returns:
        None
    """
    # Calculate the feature importances
    importances = model.feature_importances_
    # Sort the feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Arrange the feature names to match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)
    plt.close()


def train_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> None:
    """
    Train and store the model results: images + scores, and store models.
    input:
        X_train: pandas dataframe, training data features.
        X_test: pandas dataframe, testing data features.
        y_train: pandas series, training data target variable.
        y_test: pandas series, testing data target variable.
    output:
        None
    """
    # set the model Classes
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    # define the grid search space
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # fit the models
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)

    # predict random forrest on train and test data
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    # predict logistic regression on train and test data
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # generate and store the roc curve
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    _ = plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig('./images/results/roc_curve_result.png')
    plt.close()

    # save the best models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # store the model classification image results
    _classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    # store thr feature importance plot
    _feature_importance_plot(
        cv_rfc.best_estimator_,
        X_train,
        './images/results/feature_importances.png')


if __name__ == "__main__":
    # import data
    DATA = import_data("./data/bank_data.csv")

    # perform eda
    perform_eda(DATA)

    # dataset is split into train and test data
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        DATA, 'Churn')

    # train the models and store the results
    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
