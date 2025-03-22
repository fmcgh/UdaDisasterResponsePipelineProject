import sys
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from nltk.tokenize import word_tokenize
import nltk

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

nltk.download('punkt')


def load_data(database_filepath):
    """
    Loads data from a SQLite database and splits it into features (X) and target labels (Y).

    Args:
    database_filepath (str): Path to the SQLite database file.

    Returns:
    X (pd.Series): The 'message' column as the feature.
    Y (pd.DataFrame): The rest of the columns as the target labels.
    category_names (Index): The names of the categories (target labels).
    """
    conn = sqlite3.connect(database_filepath)
    
    # Load data into pandas DataFrame
    df = pd.read_sql('SELECT * FROM disaster_messages', conn)

    # The 'message' column is the feature (X)
    X = df['message']

    # The rest of the columns are the categories (Y)
    Y = df.drop(columns=['message', 'id', 'original', 'genre'])

    # Get the category names
    category_names = Y.columns

    conn.close()

    return X, Y, category_names


def tokenize(text):
    """
    Tokenizes and cleans text by lowercasing and removing non-alphabetic tokens.

    Args:
    text (str): The text to be tokenized and cleaned.

    Returns:
    words (list): A list of clean tokens.
    """
    tokens = word_tokenize(text)
    words = [word.lower() for word in tokens if word.isalpha()]
    return words


def build_model():
    """
    Builds a machine learning pipeline and wraps it with GridSearchCV for hyperparameter tuning.
    
    The pipeline consists of:
      - A TfidfVectorizer for transforming text into numerical features.
      - A MultiOutputClassifier using LogisticRegression for multi-label classification.
      
    Returns:
      cv (GridSearchCV): Grid search object with the pipeline and parameter grid.
    """
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(tokenizer=tokenize)),
        ('classifier', MultiOutputClassifier(LogisticRegression()))
    ])
    
    # Creating parameters to search
    parameters = {
        'vectorizer__max_df': [0.8, 1.0],
        'classifier__estimator__C': [1, 10] 
    }
    
    # Create a GridSearchCV object
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_weighted', cv=3, verbose=3)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the model by predicting on the test set and printing classification reports for each category.

    Args:
    model (Pipeline): The trained model.
    X_test (pd.Series): The feature data for testing.
    Y_test (pd.DataFrame): The true labels for testing.
    category_names (Index): The names of the categories to evaluate.
    """
    Y_pred = model.predict(X_test)
    
    for i, category in enumerate(category_names):
        print(f"Category: {category}")
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Saves the trained model to a file.

    Args:
    model (Pipeline): The trained model.
    model_filepath (str): The file path to save the model to.
    """
    joblib.dump(model, model_filepath)


def check_for_single_class_in_categories(Y_train):
    """
    Checks if any categories contain only one class in the training set.

    Args:
    Y_train (pd.DataFrame): The training labels.

    Returns:
    single_class_columns (list): A list of columns that have only one class.
    """
    single_class_columns = []
    for col in Y_train.columns:
        if len(Y_train[col].unique()) == 1:
            single_class_columns.append(col)
    return single_class_columns


def main():
    """
    The main entry point for the script. It loads data, trains a classifier, evaluates it, 
    and saves the trained model.

    It expects two command line arguments:
        - The file path of the disaster messages database.
        - The file path to save the trained model.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print(f'Loading data...\n    DATABASE: {database_filepath}')
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        single_class_columns = check_for_single_class_in_categories(Y_train)
        
        if single_class_columns:
            print(f"Warning: The following categories have only one class in the "
                  f"training set and will be skipped: {', '.join(single_class_columns)}")
            Y_train = Y_train.drop(columns=single_class_columns)
            Y_test = Y_test.drop(columns=single_class_columns)
            category_names = [col for col in category_names if col not in single_class_columns]

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print(f'Saving model...\n    MODEL: {model_filepath}')
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()