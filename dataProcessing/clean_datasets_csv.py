import pandas as pd
import numpy as np
import csv
import os

import pdb
import glob
import re

def separate_words_from_symbols(question):
    """ This function seperates the symbols from the words
        so that all the words or letters are isolated. This
        should help when representing the words """

    # Specific to the publishers' data -- want to do this before deleting symbols
    question = re.sub("[\<].*?[\>]", '', question) # deleting everthing between <> Quartz
    question = question.replace("\\n", "") # Guardian removing \\n
    question = re.sub("For us to continue writing great stories, we need to display ads.             Please select the extension that is blocking ads.     Please follow the steps below", "", question) # Atlantic
    question = re.sub("I want to receive updates from partners and sponsors", "", question) # Atlantic
    question = re.sub("This article is part of a feature we also send out via email as The Atlantic Daily, a newsletter with stories, ideas, and images from The Atlantic, written specially for subscribers. To sign up, please enter your email address in the field provided here", "", question) # Atlantic
    question = re.sub("This article is part of a feature we also send out via email as Politics  Policy Daily, a daily roundup of events and ideas in American politics written specially for newsletter subscribers. To sign up, please enter your email address in the field provided here", "", question) # Atlantic
    # ---

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
    question = re.sub("\/", " / ", question)
    question = re.sub("\:", " : ", question)
    question = re.sub("\£", " £ ", question)
    question = re.sub("\-", " - ", question)
    question = re.sub("\+", " + ", question)
    question = re.sub("\=", " = ", question)

    # question = re.sub("\<", " < ", question)
    # question = re.sub("\>", " > ", question)
    # question = re.sub('\"', ' " ', question)
    # question = re.sub("[\\]", " \ ", question)

    # Deleting everything exept letters and numbers
    question = re.sub('[^A-Za-z0-9]+', ' ', question)

    # Specific to the publishers' data
    question = re.sub("x9d", "", question) # Economist
    question = re.sub("x82", "", question) # Economist
    question = re.sub("xa0", "", question) # Quartz
    question = re.sub("t t t t t t t t t", "", question) # Economist
    question = re.sub("t t t t t t t t t t", "", question) # Economist
    # ----

    return question


def compile_txt_to_csv() :
    log_every = 1000         # How often do you want status updates in console?
    word_threshold = 50       # Only store article if it has a word count above the threshold

    publisher = 'Quartz'  # Datasets from which publisher are you working with?
    print('Compiling .txt files from: ' + publisher) # Write to console

    # Path to folders where raw files are stored
    data_path_read = os.path.join(publisher, 'raw', 'articles')
    # Path to folder where cleaned files are stored
    data_path_write = os.path.join(publisher, 'cleaned', 'thres_' + str(word_threshold) + '_words')
    if not os.path.exists(data_path_write): os.makedirs(data_path_write)
    # Name of .csv file being created
    csv_name = publisher + '_thres' + str(word_threshold) + '_TEMP.csv'

    # Returns a list of all files in the folder
    list_of_files = glob.glob(os.path.join(data_path_read, '*'))

    count = 0
    with open(os.path.join(data_path_write, csv_name), 'w', newline = '') as file_out :

        # Writing the header
        file_out.write('content,publication\n')

        # Reading articles from each .txt file
        for i in range(len(list_of_files)) :
            data_list = open(list_of_files[i], 'r', encoding = 'utf8').readlines()

            # Cleaning
            data_list = separate_words_from_symbols(str(data_list)) # Output is a list
            data_list = data_list.lower().strip()
            data_list = str(data_list)
            data_list = data_list[:32000]           # Making sure article fits into memory of one Excel cell (32759 characters), note that here the last word in article might be partially finished due to cutoff
            data_list = " ".join(data_list.split()) # To get rid of extra white spaces (split into words and then merge again - more difficult to write splitted sentence to file b/c of commas)


            Write article to .csv file if it is not empty (or if it has words exceeding a threshold)
            if len(data_list.split()) > word_threshold :
                file_out.write(data_list + ',' + publisher + '\n') # Note a cell in excel can only hold up to 32759 characters
                count = count + 1

            # Write status to console when 'x' articles have been read
            if (i % log_every) == 0:
                print('Reading datafile #' + str(i))

    file_out.close()

    # Printing statistics to console
    print('Total articles: ' + str(len(list_of_files)))
    print('Total cleaned: ' + str(len(pd.read_csv(os.path.join(data_path_write, csv_name)))))

    pdb.set_trace()

    # Writing statistics to file
    file = open(os.path.join(data_path_write, 'info_2.txt'), 'w')
    file.write('Number of total articles in dataset: ' + str(len(list_of_files)) + '\n')
    file.write('Number of cleaned articles in dataset: ' + str(len(pd.read_csv(os.path.join(data_path_write, csv_name)))))
    file.close

    finished = str('Run finished, .csv has been saved')

    return finished


def clean_csv() :
    log_every = 1000              # How often do you want status updates in console?
    word_threshold = 50           # Only store article if it has a word count above the threshold

    main_folder = 'Various'                 # Leave empty if .csv does not contain various publishers
    publisher_all = ['Business Insider']    # If main_folder left empty, publisher_all can only have one entry

    # Path to folders where raw files are stored
    if main_folder : # If not empty, execute
        data_path_read = os.path.join(main_folder, 'raw', 'articles')
    else :
        if len(publisher_all) != 1 : raise ValueError('Check variable publisher_all, should only have one entry')
        data_path_read = os.path.join(publisher, 'raw', 'articles')

    # Returns a list of all files in the folder
    list_of_files = glob.glob(os.path.join(data_path_read, '*'))

    # Loading all the .csv files in folder
    data_tot = pd.DataFrame()
    for i in range(len(list_of_files)) :
        data_temp = pd.read_csv(list_of_files[i])
        data_tot = data_tot.append(data_temp).reset_index(drop = True)
    data_tot = data_tot[['content', 'publication']]

    for i in range(len(publisher_all)) :

        publisher = publisher_all[i]
        # Write to console
        print('Publisher: ' + publisher)

        # Path to folder where cleaned files are stored
        data_path_write = os.path.join(publisher, 'cleaned', 'thres_' + str(word_threshold) + '_words')
        if not os.path.exists(data_path_write): os.makedirs(data_path_write)
        # Name of .csv file being created
        csv_name = publisher + '_thres' + str(word_threshold) + '.csv'

        # Obtain the data from the specific publisher
        data = data_tot[data_tot.publication == publisher].reset_index(drop = True)

        # Cleaning the data
        data_cleaned = pd.DataFrame()
        for j in range(data['content'].size) :
            data_temp = separate_words_from_symbols(data['content'][j]) # Output is a list
            data_temp = data_temp.lower().strip()
            data_temp = data_temp[:32000]           # Making sure article fits into memory of one Excel cell (32759 characters), note that here the last word in article might be partially finished due to cutoff
            data_temp = " ".join(data_temp.split()) # To get rid of extra white spaces (split into words and then merge again - more difficult to write splitted sentence to file b/c of commas)
            data['content'][j] = data_temp

            # Save row if article not empty or word count above some threshold (Note: Here we split sentence into words)
            if len(data_temp.split()) > word_threshold :
                data_cleaned = data_cleaned.append(data.iloc[j])

            # Write status to console when 'x' articles have been read
            if (j % log_every) == 0:
                print('Reading datafile #' + str(j))

        # Writing cleaned dataframe to .csv file
        data_cleaned.to_csv(os.path.join(data_path_write, csv_name), index = False)

        # Printing statistics to console
        print('Total articles: ' + str(len(data)))
        print('Total cleaned: ' + str(len(data_cleaned)))
        # Writing statistics to file
        file = open(os.path.join(data_path_write, 'info.txt'), 'w')
        file.write('Number of total articles in dataset: ' + str(len(data)) + '\n')
        file.write('Number of cleaned articles in dataset: ' + str(len(data_cleaned)))
        file.close()

    return data_cleaned


if __name__ == "__main__" :

    # data = clean_csv()

    data = compile_txt_to_csv()




