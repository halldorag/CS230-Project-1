"""Read, split and save the article dataset for our model"""
import pdb

import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import glob

import string
import csv
import os
import sys


def load_dataset(publishers, nr_articles, word_threshold) :

    # Media bias categories
    left = ['Guardian', 'Huffington Post', 'Atlantic', 'Buzzfeed News', 'CNN', 'New York Times', 'Vox', 'Washington Post', 'Economist']
    center = ['Business Insider', 'Financial Times', 'Quartz', 'Wall Street Journal', 'Reuters', 'NPR']
    right = ['Breitbart', 'Fox News', 'National Review', 'New York Post']

    data_tot = pd.DataFrame()
    for i in range(len(publishers)) :
        data_path_read = os.path.join(publishers[i], 'cleaned', 'thres_' + str(word_threshold) + '_words')
        list_of_files = glob.glob(os.path.join(data_path_read, '*.csv')) # Returns a list of all files in folder

        # If multiple .csv files available, then append all together for the publisher
        data = pd.DataFrame()
        for j in range(len(list_of_files)) :
            data_temp = pd.read_csv(list_of_files[j])
            data = data.append(data_temp).reset_index(drop = True)
        # Taking only a portion of available articles for the publisher (seed = random_state)
        data = data.sample(nr_articles[i], random_state = 1).reset_index()
        data = data[['content', 'publication']]

        # Creating labels for each publisher (e.g. New York Times: 0, Breibart: 1)
        data['label'] = ''
        if publishers[i] in left :
            data['label'] = 0
        elif publishers[i] in right :
            data['label'] = 1
        elif publishers[i] in center :
            data['label'] = 2

        # Compiling data from multiple publishers
        data_tot = data_tot.append(data).reset_index(drop = True)

    # dataset = data_tot.values
    dataset = data_tot

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

    publishers = ['New York Times', 'Guardian', 'Washington Post', 'Breitbart', 'National Review', 'New York Post']
    nr_articles = [3334, 3333, 3333, 3334, 3333, 3333]


    word_threshold = 50

    # Load the dataset into memory
    print("Loading dataset into memory...")
    dataset = load_dataset(publishers, nr_articles, word_threshold)
    print("- done.")

    # Shuffle and split the dataset into train, dev and split
    dataset2 = dataset.sample(frac = 1, random_state = 1) #.reset_index(drop=True)

    train_dataset = dataset2[:int(0.7*len(dataset2))]
    dev_dataset = dataset2[int(0.7*len(dataset2)) : int(0.85*len(dataset2))]
    test_dataset = dataset2[int(0.85*len(dataset2)):]

    trainLabels = train_dataset['label']
    devLabels = dev_dataset['label']
    testLabels = test_dataset['label']

    unique, counts = np.unique(trainLabels, return_counts = True)
    train_plot = dict(zip(unique, counts))
    if 0 in train_plot.keys() : train_plot['Left'] = train_plot.pop(0) # Renaming keys for plotting
    if 1 in train_plot.keys() : train_plot['Right'] = train_plot.pop(1)
    if 2 in train_plot.keys() : train_plot['Center'] = train_plot.pop(2)

    unique, counts = np.unique(devLabels, return_counts = True)
    dev_plot = dict(zip(unique, counts))

    unique, counts = np.unique(testLabels, return_counts = True)
    test_plot = dict(zip(unique, counts))


    # # Investigating train/dev/test distribution
    # trainLabels = np.zeros((len(train_dataset),), dtype = np.int32)
    # for i in range(len(train_dataset)) :
    #     _, trainLabels[i] = train_dataset[i]
    # unique, counts = np.unique(trainLabels, return_counts = True)
    # train_plot = dict(zip(unique, counts))
    # if 0 in train_plot.keys() : train_plot['Left'] = train_plot.pop(0) # Renaming keys for plotting
    # if 1 in train_plot.keys() : train_plot['Right'] = train_plot.pop(1)
    # if 2 in train_plot.keys() : train_plot['Center'] = train_plot.pop(2)

    # devLabels = np.zeros((len(dev_dataset),), dtype = np.int32)
    # for i in range(len(dev_dataset)) :
    #     _, devLabels[i] = dev_dataset[i]
    # unique, counts = np.unique(devLabels, return_counts = True)
    # dev_plot = dict(zip(unique, counts))

    # testLabels = np.zeros((len(test_dataset),), dtype = np.int32)
    # for i in range(len(test_dataset)) :
    #     _, testLabels[i] = test_dataset[i]
    # unique, counts = np.unique(testLabels, return_counts = True)
    # test_plot = dict(zip(unique, counts))

    fig, axs = plt.subplots(1, 3, figsize = (9, 3), sharey = True)
    axs[0].bar(list(train_plot.keys()), list(train_plot.values())), axs[0].set_title('Train set'), axs[0].set_ylabel('Frequency')
    axs[1].bar(list(train_plot.keys()), list(dev_plot.values())), axs[1].set_title('Dev set') # Note: using train keys (for simplicity)
    axs[2].bar(list(train_plot.keys()), list(test_plot.values())), axs[2].set_title('Test set') # Note: using train keys (for simplicity)
    #fig.suptitle('Train/dev/test set distributions')
    fig.savefig('Z_data/data_distribution.png')

    # Writing datasets to .csv file
    train_dataset.to_csv('Z_data/train/train.csv', index = False)
    dev_dataset.to_csv('Z_data/dev/dev.csv', index = False)
    test_dataset.to_csv('Z_data/test/test.csv', index = False)
    dataset2.to_csv('Z_data/total/total.csv', index = False)


    # Writing statistics to file
    file = open('Z_data/statistics.txt', 'w')
    file.write('Total articles in dataset: ' + str(len(dataset2)) + '\n')
    file.write('Train set: ' + str(len(train_dataset)) + '\n')
    file.write('Dev set: ' + str(len(dev_dataset)) + '\n')
    file.write('Test set: ' + str(len(test_dataset)) + '\n')
    file.write('Publishers: \n')
    for i in range(len(publishers)) :
        file.write(str(publishers[i]) + ' : ' + str(nr_articles[i]) + '\n')
    file.close()
