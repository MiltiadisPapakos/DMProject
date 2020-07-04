from tensorflow import keras
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import nltk.corpus
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


LANGUAGE = "english"


# Removes stopwords by creating a new string containing only the non-stopwords of the given string.
def remove_stopwords(string):
    words = string.split(" ")
    clear_words = [word for word in words if word not in stopwords.words(LANGUAGE)]
    return ' '.join(clear_words)


# Pre-processes the data by stemming them and removing the stopwords.
def pre_process_data(dataframe):
    df = dataframe.copy()
    nltk.download("stopwords")

    print("Started prepossessing.")

    stemmer = SnowballStemmer(LANGUAGE)
    df["text"] = pd.Series([stemmer.stem(title) for title in df["text"]])
    df["text"] = pd.Series([remove_stopwords(title) for title in df["text"]])

    print("Finished prepossessing.")

    return df


# Creates the training sets, the validation sets and the testing sets.
# It does that by taking each string and representing it using a bag of words representation with tf-idf as weight.
# 25% of the data are used for the testing set and 75% for training
# (10% of them for the validation set and the rest for the training set).
def create_training_sets(df):
    vectorizer = TfidfVectorizer()

    X_train_full, X_test, y_train_full, y_test = train_test_split(df["text"], df["label"], test_size=0.25,
                                                                  random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42)

    X_train_tf = vectorizer.fit_transform(X_train)
    X_valid_tf = vectorizer.transform(X_valid)
    X_test_tf = vectorizer.transform(X_test)

    return X_train_tf, y_train, X_valid_tf, y_valid, X_test_tf, y_test


# Trains a model give the training sets and the validation sets.
# The model uses a sigmoid function as an exit because the output is either 0 or 1.
# The training will stop after 100 epochs or when the changes in the validation set are too small.
# The metrics used to count the accuracy of the model are Precision and Recall.
def train_model(X_train_tf, y_train, X_valid_tf, y_valid):
    model = Sequential([
        layers.Dense(10, input_dim=X_train_tf.shape[1], activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=[keras.metrics.Precision(), keras.metrics.Recall()])
    print(model.summary())

    history = model.fit(X_train_tf.toarray(), y_train, epochs=100,
                        validation_data=(X_valid_tf.toarray(), y_valid),
                        callbacks=[keras.callbacks.EarlyStopping(patience=20)])

    return model


# Runs the program by loading the data from the csv file to the data set, pre-processing the data, creating the training
# sets and by training the model.
# After that it tests the training model using the testing set and it prints the accuracy metrics: Precision, Recall and
# f1 score.
def run():
    df = pd.read_csv("datasets/2/onion-or-not.csv")

    df = pre_process_data(df)
    X_train_tf, y_train, X_valid_tf, y_valid, X_test_tf, y_test = create_training_sets(df)
    model = train_model(X_train_tf, y_train, X_valid_tf, y_valid)

    loss, precision, recall = model.evaluate(X_test_tf.toarray(), y_test)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print("Precision: {:4f}".format(precision))
    print("Recall: {:4f}".format(recall))
    print("f1 score: {:4f}".format(f1_score))
