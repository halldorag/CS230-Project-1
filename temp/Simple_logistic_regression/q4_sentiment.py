#!/usr/bin/env python

import pdb

import argparse
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import itertools

from utils.treebank import StanfordSentiment
import utils.glove as glove

#from q3_sgd import load_saved_params, sgd

# We will use sklearn here because it will run faster than implementing
# ourselves. However, for other parts of this assignment you must implement
# the functions yourself!
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def getArguments():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pretrained", dest="pretrained", action="store_true",
                       help="Use pretrained GloVe vectors.")
    group.add_argument("--yourvectors", dest="yourvectors", action="store_true",
                       help="Use your vectors from q3.")
    return parser.parse_args()


def getSentenceFeatures(tokens, wordVectors, sentence):
    """
    Obtain the sentence feature for sentiment analysis by averaging its
    word vectors
    """

    # Implement computation for the sentence features given a sentence.

    # Inputs:
    # tokens -- a dictionary that maps words to their indices in
    #           the word vector list
    # wordVectors -- word vectors (each row) for all tokens
    # sentence -- a list of words in the sentence of interest

    # Output:
    # - sentVector: feature vector for the sentence

    sentVector = np.zeros((wordVectors.shape[1],))

    ### YOUR CODE HERE
    for word in sentence:
        sentVector += wordVectors[tokens[word]]
        # print(word) # HG
    sentVector /= len(sentence)
    ### END YOUR CODE

    assert sentVector.shape == (wordVectors.shape[1],)
    return sentVector


def getRegularizationValues():
    """Try different regularizations

    Return a sorted list of values to try.
    """
    values = None   # Assign a list of floats in the block below
    ### YOUR CODE HERE
    values = [10**i for i in np.random.uniform(-5,0,10)]
    ### END YOUR CODE
    return sorted(values)


def chooseBestModel(results):
    """Choose the best model based on parameter tuning on the dev set

    Arguments:
    results -- A list of python dictionaries of the following format:
        {
            "reg": regularization,
            "clf": classifier,
            "train": trainAccuracy,
            "dev": devAccuracy,
            "test": testAccuracy
        }

    Returns:
    Your chosen result dictionary.
    """
    bestResult = None

    ### YOUR CODE HERE
    best_dev_acc = 0
    for model in results:
        if model['dev'] > best_dev_acc:
            bestResult = model
    ### END YOUR CODE

    return bestResult


def accuracy(y, yhat):
    """ Precision for classifier """
    assert(y.shape == yhat.shape)
    return np.sum(y == yhat) * 100.0 / y.size


def plotRegVsAccuracy(regValues, results, filename):
    """ Make a plot of regularization vs accuracy """
    plt.plot(regValues, [x["train"] for x in results])
    plt.plot(regValues, [x["dev"] for x in results])
    plt.xscale('log')
    plt.xlabel("regularization")
    plt.ylabel("accuracy")
    plt.legend(['train', 'dev'], loc='upper left')
    plt.savefig('results/' + filename)


def outputConfusionMatrix(features, labels, clf, filename):
    """ Generate a confusion matrix """
    pred = clf.predict(features)
    cm = confusion_matrix(labels, pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.colorbar()
    classes = ["Left", "Center","Right"]
    #classes = ["- -", "-", "neut", "+", "+ +"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 1.1#2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    #plt.tight_layout()
    #plt.margins(0.2)
    plt.subplots_adjust(bottom=0.15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('results/' + filename)

    return cm


def outputPredictions(dataset, features, labels, clf, filename):
    """ Write the predictions to file """
    pred = clf.predict(features)
    with open('results/' + filename, "w") as f:
        print >> f, "True\tPredicted\tText"
        for i in xrange(len(dataset)):
            print >> f, "%d\t%d\t%s" % (
                labels[i], pred[i], " ".join(dataset[i][0]))


def main(args):
    """ Train a model to do sentiment analysis"""

    # Load the dataset
    dataset = StanfordSentiment()
    print('Preparing tokens...')
    tokens = dataset.tokens()
    nWords = len(tokens)
    print('-done preparing tokens.')

    if args.yourvectors:
        pass
        # _, wordVectors, _ = load_saved_params()
        # wordVectors = np.concatenate(
        #     (wordVectors[:nWords,:], wordVectors[nWords:,:]),
        #     axis=1)
    elif args.pretrained:
        wordVectors = glove.loadWordVectors(tokens)
    dimVectors = wordVectors.shape[1]


    # Loading and preparing the train set
    print('Loading and preparing train/dev/test sets ...')
    d = []
    with open("data/articles/train/sentences.txt") as f :
        for line in f :
            d.append(line.split())

    trainFeatures = np.zeros((len(d), dimVectors))
    trainLabels = np.loadtxt('data/articles/train/labels.txt', dtype = 'int32') #.reshape(1, len(d))
    for i in range(0,100) : #range(len(d)) :
        # print(len(d[i]))
        # print(i)
        trainFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, d[i])
    trainset = zip(d, trainLabels) # A tuple of words (list, splitted sentences) and labels (array of integers)

    # pdb.set_trace()

    # Loading and preparing the dev set
    d = []
    with open("data/articles/dev/sentences.txt") as f :
        for line in f :
            d.append(line.split())

    devFeatures = np.zeros((len(d), dimVectors))
    devLabels = np.loadtxt('data/articles/dev/labels.txt', dtype = 'int32') #.reshape(1, len(d))
    for i in range(len(d)):
        devFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, d[i])
    devset = zip(d, devLabels) # A tuple of words (list, splitted sentences) and labels (array of integers)

    # Loading and preparing the test set
    d = []
    with open("data/articles/test/sentences.txt") as f :
        for line in f :
            d.append( line.split())

    testFeatures = np.zeros((len(d), dimVectors))
    testLabels = np.loadtxt('data/articles/test/labels.txt', dtype = 'int32') #.reshape(1, len(d))
    for i in range(len(d)):
        testFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, d[i])
    testset = zip(d, testLabels) # A tuple of words (list, splitted sentences) and labels (array of integers)

    print('-done preparing train/dev/test sets.')

    # pdb.set_trace()

    # We will save our results from each run
    results = []
    regValues = getRegularizationValues()
    for reg in regValues:
        print "Training for reg=%f" % reg
        # Note: add a very small number to regularization to please the library
        clf = LogisticRegression(C=1.0/(reg + 1e-12))
        clf.fit(trainFeatures, trainLabels)

        # Test on train set
        pred = clf.predict(trainFeatures)
        trainAccuracy = accuracy(trainLabels, pred)
        print "Train accuracy (%%): %f" % trainAccuracy

        # Test on dev set
        pred = clf.predict(devFeatures)
        devAccuracy = accuracy(devLabels, pred)
        print "Dev accuracy (%%): %f" % devAccuracy

        # Test on test set
        # Note: always running on test is poor style. Typically, you should
        # do this only after validation.
        pred = clf.predict(testFeatures)
        testAccuracy = accuracy(testLabels, pred)
        print "Test accuracy (%%): %f" % testAccuracy

        results.append({
            "reg": reg,
            "clf": clf,
            "train": trainAccuracy,
            "dev": devAccuracy,
            "test": testAccuracy})

    # Print the accuracies
    print ""
    print "=== Recap ==="
    print "Reg\t\tTrain\tDev\tTest"
    for result in results:
        print "%.2E\t%.3f\t%.3f\t%.3f" % (
            result["reg"],
            result["train"],
            result["dev"],
            result["test"])
    print ""

    bestResult = chooseBestModel(results)
    print "Best regularization value: %0.2E" % bestResult["reg"]
    print "Test accuracy (%%): %f" % bestResult["test"]

    # do some error analysis
    if args.pretrained:
        plotRegVsAccuracy(regValues, results, "q4_reg_v_acc.png")
        cm = outputConfusionMatrix(devFeatures, devLabels, bestResult["clf"],
                              "q4_dev_conf.png")
        outputPredictions(devset, devFeatures, devLabels, bestResult["clf"],
                          "q4_dev_pred.txt")

    pdb.set_trace()


if __name__ == "__main__":
    main(getArguments())
