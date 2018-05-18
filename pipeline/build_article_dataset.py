"""Read, split and save the kaggle dataset for our model"""

import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

import string
import csv
import os
import sys

def separate_words_from_symbols(question):
    """ This function seperates the symbols from the words
	    so that all the words or letters are isolated. This
	    should help when representing the words """
    question = re.sub("\?", " ? ", question)
    question = re.sub("\!", " ! ", question)
    #question = re.sub("\'", " ' ", question)
    question = re.sub("\,", " , ", question)
    question = re.sub("\.", " . ", question)
    question = re.sub("\-", " - ", question)
    question = re.sub("\)", " ) ", question)
    question = re.sub("\(", " ( ", question)
    question = re.sub(" \” ", " ” ", question)
    question = re.sub("\“", " “ ", question)
    #question = re.sub('\"', ' " ', question)
    question = re.sub("\/", " / ", question)
    question = re.sub("\:", " : ", question)
    question = re.sub("\£", " £ ", question)
    # question = re.sub("[\\]", " \ ", question)

    # Deleting everything exept letters and numbers
    question = re.sub('[^A-Za-z0-9]+', ' ', question)

    return question


def load_dataset(path_csv):
    """Loads dataset into memory from csv file"""
    # Open the csv file, need to specify the encoding for python3
    use_python3 = sys.version_info[0] >= 3
    with (open(path_csv, encoding="windows-1252") if use_python3 else open(path_csv)) as f:

        publishers = ['New York Times', 'Guardian', 'Washington Post', 'Breitbart', 'National Review', 'New York Post', 'Reuters', 'NPR']
        # publishers = ['New York Times', 'Guardian', 'Breitbart', 'National Review', 'Reuters', 'NPR']

        # Media bias categories
        left = ['Guardian', 'Huffington Post', 'Atlantic', 'Buzzfeed News', 'CNN', 'New York Times', 'Vox', 'Washington Post']
        center = ['Business Insider', 'Financial Times', 'Quartz', 'Wall Street Journal', 'Reuters', 'NPR']
        right = ['Breitbart', 'Fox News', 'National Review', 'New York Post']

        data1 = pd.read_csv('../../Data/Datasets/articles1.csv')
        data2 = pd.read_csv('../../Data/Datasets/articles2.csv')
        data3 = pd.read_csv('../../Data/Datasets/articles3.csv')
        data = pd.concat([data1, data2, data3])

        data = data[['content', 'publication']]

        # Creating an empty dataframe
        data_filtered = pd.DataFrame()
        for i in range(len(publishers)):

            # To ensure equal categories, cut off larger sources
            if publishers == 'Reuters' or publishers == 'NPR' :
                cut_off = 100000
            else :
                cut_off = 8000
            data_filtered = data_filtered.append(data[data.publication == publishers[i]][0:cut_off]).reset_index(drop = True)

            # Creating labels for each publisher (e.g. New York Times: 0, Breibart: 1)
            if publishers[i] in left :
                data_filtered.replace([publishers[i]], -1, inplace = True)
            elif publishers[i] in center :
                data_filtered.replace([publishers[i]], 0, inplace = True)
            elif publishers[i] in right :
                data_filtered.replace([publishers[i]], 1, inplace = True)

        # Cleaning data
        # Separating words from symbols and removing all symbols except letters and numbers
        a = pd.DataFrame()
        for j in range(data_filtered['content'].size) :
            #print(j)
            a = a.append(pd.Series(separate_words_from_symbols(data_filtered['content'][j])), ignore_index = True)
        data_filtered['content'] = a
        # Ensuring lower case letters, removing white spaces at beginning and end of sentences, splitting sentences into strings of words
        data_filtered['content'] = data_filtered['content'].str.lower() # Ensuring only lower case letters
        data_filtered['content'] = data_filtered['content'].str.strip() # Removing white space at beginning and end of sentence
        data_filtered['content'] = data_filtered['content'].str.split() # Splitting sentences into strings of words
        # data_filtered['content'] = data_filtered['content'].str.replace('[{}]'.format(string.punctuation), '') # Removing punctiation (to see which ones print string.punctuation)

        # Changing the format to a tuple for further analysis
        dataset = [tuple(x) for x in data_filtered.values]

        # Deleting rows from dataset that have no content: Note that script might take a long time with all these for-loops, probably best to change this
        idx = []
        for i in range(len(dataset)) :
            test_row = dataset[i][0]
            if test_row :
                pass
            else :
                idx.append(i)
        dataset = [i for j, i in enumerate(dataset) if j not in idx]


    return dataset


def save_dataset(dataset, save_dir):
    """Writes sentences.txt and labels.txt files in save_dir from dataset

    Args:
        dataset: ([(["a", "cat"], ["O", "O"]), ...])
        save_dir: (string)
    """
    # Create directory if it doesn't exist
    print("Saving in {}...".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Export the dataset
    with open(os.path.join(save_dir, 'sentences.txt'), 'w') as file_sentences:
        with open(os.path.join(save_dir, 'labels.txt'), 'w') as file_labels:
            for content, publication in dataset:
                file_sentences.write("{}\n".format(" ".join(content)))
                # file_labels.write("{}\n".format(" ".join(publication)))
                file_labels.write("{}\n".format(publication))
    print("- done.")


if __name__ == "__main__":
    # Check that the dataset exists (you need to make sure you haven't downloaded the `ner.csv`)
    path_dataset = '../../Data/Datasets/articles1.csv'
    msg = "{} file not found. Make sure you have downloaded the right dataset".format(path_dataset)
    assert os.path.isfile(path_dataset), msg

    # Load the dataset into memory
    print("Loading dataset into memory...")
    dataset = load_dataset(path_dataset)
    print("- done.")

    # Taking only a subset of the data
    # dataset = dataset[0:40]

    # Shuffle and split the dataset into train, dev and split
    randID = np.arange(len(dataset))
    np.random.seed(seed = 1)
    np.random.shuffle(randID)
    dataset2 = list(dataset[i] for i in randID)

    train_dataset = dataset2[:int(0.7*len(dataset2))]
    dev_dataset = dataset2[int(0.7*len(dataset2)) : int(0.85*len(dataset2))]
    test_dataset = dataset2[int(0.85*len(dataset2)):]

    # Investigating train/dev/test distribution
    trainLabels = np.zeros((len(train_dataset),), dtype = np.int32)
    for i in range(len(train_dataset)) :
        _, trainLabels[i] = train_dataset[i]
    unique, counts = np.unique(trainLabels, return_counts = True)
    train_plot = dict(zip(unique, counts))
    if -1 in train_plot.keys() : train_plot['Left'] = train_plot.pop(-1) # Renaming keys for plotting
    if 0 in train_plot.keys() : train_plot['Center'] = train_plot.pop(0)
    if 1 in train_plot.keys() : train_plot['Right'] = train_plot.pop(1)

    devLabels = np.zeros((len(dev_dataset),), dtype = np.int32)
    for i in range(len(dev_dataset)) :
        _, devLabels[i] = dev_dataset[i]
    unique, counts = np.unique(devLabels, return_counts = True)
    dev_plot = dict(zip(unique, counts))

    testLabels = np.zeros((len(test_dataset),), dtype = np.int32)
    for i in range(len(test_dataset)) :
        _, testLabels[i] = test_dataset[i]
    unique, counts = np.unique(testLabels, return_counts = True)
    test_plot = dict(zip(unique, counts))

    fig, axs = plt.subplots(1, 3, figsize = (9, 3), sharey = True)
    axs[0].bar(list(train_plot.keys()), list(train_plot.values())), axs[0].set_title('Train set'), axs[0].set_ylabel('Frequency')
    axs[1].bar(list(train_plot.keys()), list(dev_plot.values())), axs[1].set_title('Dev set') # Note: using train keys (for simplicity)
    axs[2].bar(list(train_plot.keys()), list(test_plot.values())), axs[2].set_title('Test set') # Note: using train keys (for simplicity)
    #fig.suptitle('Train/dev/test set distributions')
    fig.savefig('data/articles/data_distribution.png')

    # Save the datasets to files
    save_dataset(train_dataset, 'data/articles/train')
    save_dataset(dev_dataset, 'data/articles/dev')
    save_dataset(test_dataset, 'data/articles/test')
    save_dataset(dataset2, 'data/articles/total')






