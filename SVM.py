import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.impute import KNNImputer


# Drop pH column
def drop_column(input_df):
    input_df = input_df.drop(columns='pH')
    return input_df


# Fill 33% of values in pH column with average
def average_fill(input_df):
    y = input_df.loc[:, 'pH']
    dummy = np.random.choice([0, 1], size=len(y), p=[1. / 3, 2. / 3])
    input_df['pH'] = input_df['pH'] * dummy
    input_df['pH'] = input_df['pH'].mask(input_df['pH'] == 0).fillna(input_df['pH'].mean())
    return input_df


# Predicting 33% of values with a Logostic Regression model and splitting the Dataframe with Sklearn
def log_regr(input_df):
    input_df['pH'] = input_df['pH'] * 100
    k = input_df.loc[:, 'pH']
    X_pH_train_dummy, X_pH_test, y_pH_train, y_pH_test = train_test_split(input_df, k, test_size=0.33, random_state=42)
    X_pH_train = X_pH_train_dummy.drop(columns='pH')
    X_pH_test = X_pH_test.drop(columns='pH')

    logmodel = LogisticRegression()
    logmodel.fit(X_pH_train, y_pH_train)
    predictions = logmodel.predict(X_pH_test)
    X_pH_test['pH'] = predictions
    final = X_pH_train_dummy.append(X_pH_test, ignore_index=True)
    final['pH'] = final['pH'] / 100
    return final


# Predicting  33% of values with KNNImputer which fills the Nan values with the mean of the nearest n_neighbors found in
# the training set
def k_means(input_df):
    input_df['pH'] = input_df['pH'] * 100
    k = input_df.loc[:, 'pH']
    X_pH_train, X_pH_test, y_pH_train, y_pH_test = train_test_split(input_df, k, test_size=0.33, random_state=42)
    X_pH_test = X_pH_test.drop(columns='pH')
    final = X_pH_train.append(X_pH_test, ignore_index=True)
    imputer = KNNImputer(n_neighbors=5, weights='uniform')
    np.set_printoptions(suppress=True)
    final = imputer.fit_transform(final)
    df = pd.DataFrame(final)
    df.columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                  'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
    df['pH'] = df['pH'] / 100
    return df


# Runs the program by loading the data from the csv file and using the SVM classifier.
# Depending on the input parameter method it will manipulate the dataset differently. Specifically:
# method = 0 --> do nothing (part A)
# method = 1 --> drop the pH column (part B.1)
# method = 2 --> fill pH missing values with average (part B.2)
# method = 3 --> fill pH missing values using Logistic Regression (part B.3)
# method = 4 --> fill pH missing values using k-means mean (part B.4)
def run(method):
    # Reading the csv file via absolute path
    input_df = pd.read_csv('datasets/1/winequality-red.csv')
    # Labeling columns
    input_df.columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                        'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
                        'quality']

    if method == 0:
        input_df = input_df  # change nothing
    elif method == 1:
        input_df = drop_column(input_df)
    elif method == 2:
        input_df = average_fill(input_df)
    elif method == 3:
        input_df = log_regr(input_df)
    elif method == 4:
        input_df = k_means(input_df)
    else:
        print("Incorrect input.")
        return

        # Predicting Quality column and printing classicication report with the metrics required
    x = input_df.loc[:, 'fixed acidity': 'alcohol']
    y = input_df.loc[:, 'quality']
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    clf = svm.SVC(gamma=0.001, C=100)
    clf.fit(X_train, Y_train)
    predict_quality = clf.predict(X_test)

    print(classification_report(Y_test, predict_quality))
