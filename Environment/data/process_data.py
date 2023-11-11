import sys
import pandas as pd 
import numpy as np
from sqlalchemy import create_engine

database_filepath  = "data/DisasterPipeline.db"
messages_filepath = "data/disaster_messages.csv"
categories_filepath = "data/disaster_categories.csv"


def load_data(messages_filepath, categories_filepath):
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on = 'id')
    
    return df


def clean_data(df):
    
    #Splitting values in categories column
    categories = df['categories'].str.split(';',expand=True)
    
    #Taking the first row to get column names
    row = categories.iloc[0]  
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        # setting each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
        
    #Dropping original categories column and concatting new columns to df
    df.drop('categories', inplace=True, axis=1)
    df = pd.concat([df,categories], axis = 1)
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filepath):

    engine = create_engine(f"sqlite:///{database_filepath}")
    df.to_sql("DisasterTable", engine, if_exists="replace", index=False)
    
    print(f"Data successfully saved to {database_filepath}")

def main():

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')


if __name__ == '__main__':
    main()