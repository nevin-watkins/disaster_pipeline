import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Input:
    1. messages_filepath (csv where the messages are stored)
    2. categories_filepath (csv where the categories are stored)
    
    Output:
    1. df: merged dataframe of messages and category on id
    '''
    #load messages and categories and merge on "id"
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")
    return df

def clean_data(df):
    '''
    Input: merged dataframe of categories and messages
    
    Output: cleaned dataframe
    '''
    categories = df.categories.str.split(pat=";", expand=True)
    categories.columns = [col[:-2] for col in categories.iloc[0]]
    for col in categories:
        categories[col] = categories[col].str[-1:]
        categories[col] = pd.to_numeric(categories[col])
        categories[col] = categories[col].apply(lambda x:1 if x>1 else x)
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    '''
    Input:
    1. df: cleaned dataframe
    2. database_filename: filename where we will store our clean data
    
    Saves function to sqllite db
    '''
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages_categories', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
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