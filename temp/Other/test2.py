# This is a modified script of test.py to be inserted into the tensorflow pipeline

import pandas as pd
import string

def reading_articles(FileName, publishers):

    data = pd.read_csv(FileName)
    data = data[['content', 'publication']]

    # Creating an empty dataframe
    data_filtered = pd.DataFrame()
    for i in range(len(publishers)):

        data_filtered = data_filtered.append(data[data.publication == publishers[i]])
        # Creating labels for each publisher (e.g. New York Times: 0, Breibart: 1)
        data_filtered.replace([publishers[i]], i, inplace = True)

    # Cleaning data
    data_filtered['content'] = data_filtered['content'].str.replace('[{}]'.format(string.punctuation), '') # Removing punctiation (to see which ones print string.punctuation)
    data_filtered['content'] = data_filtered['content'].str.lower() # Ensuring only lower case letters
    data_filtered['content'] = data_filtered['content'].str.strip() # Removing white space at beginning and end of sentence
    data_filtered['content'] = data_filtered['content'].str.split() # Splitting sentences into strings of words

    # Changing the format to a tuple for further analysis
    dataset = [tuple(x) for x in data_filtered.values]

    return dataset

# Additional things:
# ' '.join(data_filtered['content'].str.split())
