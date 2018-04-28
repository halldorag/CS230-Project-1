import pandas as pd
import numpy as np
import string
import re

# 0,id,title,publication,author,date,year,month,url,content


def reading_articles(FileName, publishers):
    
    data = pd.read_csv(FileName)
    df = data.set_index("id", drop = False)
    frames = []
    df_filtered = []
    
    for i in range(len(publishers)):
        
        df_filtered.append(df[(df.publication == publishers[i])])
        
        
    DataFrameFinal = pd.concat(df_filtered)
        
    return DataFrameFinal



def creating_labels(data, publishers):
    for i in range(len(publishers)):
        
        data['publication'].replace([publishers[i]], i,inplace=True)
        


FileName = 'D:\\CS230_Project\\Data\\all-the-news\\articles1.csv'

publishers = ['New York Times', 'Breitbart']

# reading the articles for certain publishers
data = reading_articles(FileName, publishers)

# creating a label for all articles based on the publisher
Y = data.copy()
creating_labels(Y, publishers)

#df_clean = data['content'].str.replace('[{}]'.format(string.punctuation), '')


df_clean = df_clean.str.lower()
df_clean = df_clean.str.strip()

df_clean = df_clean.str.split()