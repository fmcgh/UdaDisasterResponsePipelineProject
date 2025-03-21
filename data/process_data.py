import sys
import sqlite3
import pandas as pd


def load_data(messages_filepath, categories_filepath):
    """
    Loads and merges the messages and categories datasets into a single DataFrame.

    Args:
    messages_filepath (str): Path to the messages dataset (CSV).
    categories_filepath (str): Path to the categories dataset (CSV).

    Returns:
    pd.DataFrame: Merged DataFrame containing messages and categories.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge datasets on the 'id' column
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """
    Cleans the merged DataFrame by splitting the 'categories' column into separate category columns,
    converting category values to numeric, and removing duplicates.

    Args:
    df (pd.DataFrame): Merged DataFrame containing messages and categories.

    Returns:
    pd.DataFrame: Cleaned DataFrame with separate category columns and numeric values.
    """
    # Split the 'categories' column into separate category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Get the category names from the first row of the categories DataFrame
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])  # Remove last 2 characters (e.g., '1' or '0')
    categories.columns = category_colnames

    # Convert category values to numeric (1 or 0)
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1]).astype(int)

    # Drop the original 'categories' column from the DataFrame
    df.drop('categories', axis=1, inplace=True)

    # Concatenate the original dataframe with the new category columns
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Saves the cleaned DataFrame to a SQLite database.

    Args:
    df (pd.DataFrame): Cleaned DataFrame to be saved.
    database_filename (str): Path to the SQLite database file.
    """
    # Create a connection to the database
    conn = sqlite3.connect(database_filename)
    
    # Save DataFrame to database, replace any existing data in the table
    df.to_sql('disaster_messages', conn, if_exists='replace', index=False)
    
    conn.close()


def main():
    """
    Main function that loads, cleans, and saves data to a database.
    
    The function performs the following steps:
    1. Loads the messages and categories datasets.
    2. Cleans the data by splitting the categories and converting them to numeric values.
    3. Saves the cleaned data to a SQLite database.

    Command-line arguments:
    - messages_filepath: Path to the messages dataset (CSV).
    - categories_filepath: Path to the categories dataset (CSV).
    - database_filepath: Path to the SQLite database to save the cleaned data.
    """
    if len(sys.argv) == 4:
        # Extract file paths from command-line arguments
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, categories_filepath))
        # Load and merge data
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        # Clean the data
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        # Save the cleaned data to a database
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()