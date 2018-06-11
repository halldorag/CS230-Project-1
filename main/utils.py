import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import json
import pdb
import os

from sklearn.metrics import roc_curve, auc

def plot_roc(nr_classes, Y_dev, Y_dev_hat, args) :
    if nr_classes == 2 :
        fpr, tpr, thresholds_rf = roc_curve(Y_dev, Y_dev_hat)
        auc_keras = auc(fpr, tpr)
        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label = 'Validation set (area = {:.3f})'.format(auc_keras))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.savefig(os.path.join(args.model_dir, 'roc.png'))

        df = pd.DataFrame({"False postitive rates" : fpr, "True positive rates" : tpr})
        df.to_csv(os.path.join(args.model_dir, 'roc.csv'), index = False)


def error_analysis(train_dataset, Y_train_hat, dev_dataset, Y_dev_hat, test_dataset, Y_test_hat) :

    train_dataset['predicted label'] = Y_train_hat
    train_dataset = train_dataset.sort_values('publication')
    train_dataset_misclass = train_dataset[train_dataset['label'] != train_dataset['predicted label']]
    train_dataset_misclass = train_dataset_misclass.sort_values('publication')
    train_publ_misclass = train_dataset_misclass['publication'].unique().tolist()
    train_error_results = pd.DataFrame()
    for i in range(len(train_publ_misclass)) :
        temp = train_dataset_misclass.loc[train_dataset_misclass['publication'] == train_publ_misclass[i]]
        # train_error_results[train_publ_misclass[i]] = temp.groupby('predicted label')['content'].nunique()
        d = np.zeros((3))
        d[np.sort(temp['predicted label'].unique())] = temp['predicted label'].value_counts().sort_index()
        train_error_results[train_publ_misclass[i]] = d

    dev_dataset['predicted label'] = Y_dev_hat
    dev_dataset = dev_dataset.sort_values('publication')
    dev_dataset_misclass = dev_dataset[dev_dataset['label'] != dev_dataset['predicted label']]
    dev_dataset_misclass = dev_dataset_misclass.sort_values('publication')
    dev_publ_misclass = dev_dataset_misclass['publication'].unique().tolist()
    dev_error_results = pd.DataFrame()
    for i in range(len(dev_publ_misclass)) :
        temp = dev_dataset_misclass.loc[dev_dataset_misclass['publication'] == dev_publ_misclass[i]]
        # dev_error_results[dev_publ_misclass[i]] = temp.groupby('predicted label')['content'].nunique()
        d = np.zeros((3))
        d[np.sort(temp['predicted label'].unique())] = temp['predicted label'].value_counts().sort_index()
        dev_error_results[dev_publ_misclass[i]] = d

    test_dataset['predicted label'] = Y_test_hat
    test_dataset = test_dataset.sort_values('publication')
    test_dataset_misclass = test_dataset[test_dataset['label'] != test_dataset['predicted label']]
    test_dataset_misclass = test_dataset_misclass.sort_values('publication')
    test_publ_misclass = test_dataset_misclass['publication'].unique().tolist()
    test_error_results = pd.DataFrame()
    for i in range(len(test_publ_misclass)) :
        temp = test_dataset_misclass.loc[test_dataset_misclass['publication'] == test_publ_misclass[i]]
        d = np.zeros((3))
        d[np.sort(temp['predicted label'].unique())] = temp['predicted label'].value_counts().sort_index()
        test_error_results[test_publ_misclass[i]] = d

    return train_dataset_misclass, train_error_results, dev_dataset_misclass, dev_error_results, test_dataset_misclass, test_error_results


def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4).

    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this.

    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """

    m = X.shape[0]                                   # number of training examples

    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m, max_len))

    for i in range(m):                               # loop over training examples

        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        # sentence_words =[w.lower() for w in X[i].split()] # NOTE, already splitted when loading
        sentence_words =[w.lower() for w in X[i]]

        # Initialize j to 0
        j = 0

        # Loop over the words of sentence_words
        for w in sentence_words:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            if w in word_to_index:
                X_indices[i, j] = word_to_index[w]
                # Increment j to j + 1
                j = j + 1

            else:

                X_indices[i, j] = 0

                #if token not in tokens:
                #continue

    return X_indices

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.subplots_adjust(bottom=0.15) # Otherwise xlabel is cut-off

def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding = 'utf8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
