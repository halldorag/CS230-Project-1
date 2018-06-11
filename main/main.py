import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import time
import os
import argparse

from keras import regularizers
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Conv1D, MaxPooling1D, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import glorot_uniform
from keras.optimizers import Adam
from keras.callbacks import CSVLogger
from keras.utils import plot_model

from sklearn.metrics import confusion_matrix

from contextlib import redirect_stdout

from utils import *

np.random.seed(1)
start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/exp_1', help = "Directory containing params.json")
parser.add_argument('--data_dir', default='data/dataset_1', help = "Directory containing the dataset")


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """

    vocab_len = len(word_to_index) + 1                  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["cucumber"].shape[0]      # define dimensionality of your GloVe word vectors (= 50)

    ### START CODE HERE ###
    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len, emb_dim))

    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct output/input sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False.
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)
    ### END CODE HERE ###

    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))

    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


def model_fn(input_shape, word_to_vec_map, word_to_index, params, nr_classes) :
    """
    Function creating the Emojify-v2 model's graph.

    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)
    params -- parameters imported from a json file (e.g. learning rate)
    nr_classes -- number of classes in train/dev/test set

    Returns:
    model -- a model instance in Keras
    """

    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices = Input(input_shape, dtype='int32')

    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)

    if params.dropout != 0 :
        embeddings = Dropout(params.dropout)(embeddings)

    if params.model_version == 'base' :
        X_input = Input(shape = (50, ))
        X = X_input
        for i in range(params.num_layers):
            X = Dense(params.num_units)(X)
            X = Dropout(params.dropout)(X)
            X = Activation('relu')(X)
        X = Dense(nr_classes)(X)
        X = Activation('softmax')(X)

    if params.model_version == 'lstm_1L' :
        # 1st hidden layer
        X = LSTM(params.num_units, return_sequences = False)(embeddings)
        if params.dropout != 0 :
            X = Dropout(params.dropout)(X)
        # Output layer with softmax activation (to get back a batch of XD vector)
        X = Dense(nr_classes)(X)
        X = Activation('softmax')(X)

    if params.model_version == 'lstm_2L' :
        # 1st hidden layer
        X = LSTM(params.num_units, return_sequences = True)(embeddings)
        if params.dropout != 0 :
            X = Dropout(params.dropout)(X)
        # 2nd hidden layer
        X = LSTM(params.num_units, return_sequences = False)(X)
        if params.dropout != 0 :
            X = Dropout(params.dropout)(X)
        # Output layer with softmax activation (to get back a batch of XD vector)
        X = Dense(nr_classes)(X)
        X = Activation('softmax')(X)

    if params.model_version == 'conv1d' :

        # ldery@stanford
        #print("Yes I am in here")
        print("Embeddings" + str(embeddings.shape))

        l_cov1= Conv1D(params.num_units, 5, activation='relu')(embeddings)
        print("l_cov1: " + str(l_cov1.shape))

        l_pool1 = MaxPooling1D(pool_size = 5, strides = 3)(l_cov1)
        print("l_pool1: " + str(l_pool1.shape))

        l_cov2 = Conv1D(params.num_units, 5, activation='relu')(l_pool1)
        print("l_cov2: " + str(l_cov2.shape))

        l_pool2 = MaxPooling1D(pool_size = 5, strides = 3)(l_cov2)
        print("l_pool2: " + str(l_pool2.shape))

        l_cov3 = Conv1D(params.num_units, 5, activation='relu')(l_pool2)
        print("l_cov3: " + str(l_cov3.shape))

        if l_cov3.shape[1] < 36 :
            l_pool3 = MaxPooling1D(3)(l_cov3)
        else :
            l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling
        print("l_pool3: " + str(l_pool3.shape))

        l_flat = Flatten()(l_pool3)
        print("l_flat: " + str(l_flat.shape))

        X = Dense(128, activation='relu')(l_flat) # dense
        if params.dropout != 0 :
            X = Dropout(params.dropout)(X)

        # print("l_dense: " + str(l_dense.shape))
        X = Dense(nr_classes, activation='softmax')(X)

    if params.model_version == 'conv1dpluslstm' :
        print('Running conv1dpluslstm')
        # ldery@stanford
        #print("Yes I am in here")
        # print("Embeddings" + str(embeddings.shape))

        l_cov1= Conv1D(128, 10, strides=1, activation='relu')(embeddings)
        print("l_cov1: " + str(l_cov1.shape))

        l_pool1 = MaxPooling1D(2)(l_cov1)
        print("l_pool1: " + str(l_pool1.shape))

        l_cov2 = Conv1D(128, 10, activation='relu')(l_pool1)
        print("l_cov2: " + str(l_cov2.shape))

        l_pool2 = MaxPooling1D(2)(l_cov2)
        print("l_pool2: " + str(l_pool2.shape))

        X = LSTM(params.num_units, return_sequences = False)(l_pool1)
        if params.dropout != 0 :
            X = Dropout(params.dropout)(X)
        # Output layer with softmax activation (to get back a batch of XD vector)
        X = Dense(nr_classes)(X)
        X = Activation('softmax')(X)


    # Create Model instance which converts sentence_indices into X.
    if params.model_version == 'base' :
        model = Model(inputs = X_input, outputs = X)
    else :
        model = Model(inputs = sentence_indices, outputs = X)

    return model


if __name__ == '__main__':

    args = parser.parse_args()
    # Printing to console
    print('Running: ' + args.model_dir)

    # Setting path to experiments and data directory
    markus = 'no' # If Markus is running on Linux
    path_markus = '/home/ubuntu/Project/'
    if markus == 'yes' :
        args.model_dir = os.path.join(path_markus, args.model_dir)
        args.data_dir = os.path.join(path_markus, args.data_dir)

    # Loading base model parameters
    params = Params(os.path.join(args.model_dir, 'params.json'))

    # Loading train/dev/test sets
    train_dataset = pd.read_csv(args.data_dir + '/train/train.csv')
    X_train = np.asarray(train_dataset['content'].str.split())
    Y_train = np.asarray(train_dataset['label'], dtype = 'int32')

    dev_dataset = pd.read_csv(args.data_dir + '/dev/dev.csv')
    X_dev = np.asarray(dev_dataset['content'].str.split())
    Y_dev = np.asarray(dev_dataset['label'], dtype = 'int32')

    test_dataset = pd.read_csv(args.data_dir + '/test/test.csv')
    X_test = np.asarray(test_dataset['content'].str.split())
    Y_test = np.asarray(test_dataset['label'], dtype = 'int32')

    # Number of classes in dataset
    nr_classes = max(len(np.unique(Y_train)), len(np.unique(Y_dev)), len(np.unique(Y_test))) # Throw an error if not all labels in all sets

    #--- NOTE: Comment out for the actual run (do some overfitting (10 training samples))
    # nr_temp = 20
    # Y_train = Y_train[:nr_temp,]
    # X_train = X_train[:nr_temp,]
    # train_dataset = train_dataset[:nr_temp]
    # Y_dev = Y_dev[:nr_temp,]
    # X_dev = X_dev[:nr_temp,]
    # dev_dataset = dev_dataset[:nr_temp]
    # Y_test = Y_test[:nr_temp,]
    # X_test = X_test[:nr_temp,]
    # test_dataset = test_dataset[:nr_temp]
    #---

    # Converting labels to one hot vector
    Y_train_oh = convert_to_one_hot(Y_train, C = nr_classes)
    Y_dev_oh = convert_to_one_hot(Y_dev, C = nr_classes)
    Y_test_oh = convert_to_one_hot(Y_test, C = nr_classes)

    # Maximum word count of a sentence
    maxTrain = len(max(X_train, key=len))
    maxDev = len(max(X_dev, key=len))
    maxTest = len(max(X_test, key=len))
    maxLen = max(maxTrain, maxDev, maxTest)

    # Vector representation of words (using GloVe)
    word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(os.path.join(os.path.dirname(args.data_dir), 'glove.6B.50d.txt'))
    # Test set
    X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
    X_train_indices = X_train_indices[:,: params.num_words_in_article]
    # Train set
    X_dev_indices = sentences_to_indices(X_dev, word_to_index, maxLen)
    X_dev_indices = X_dev_indices[:,: params.num_words_in_article]
    # Dev set
    X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
    X_test_indices = X_test_indices[:,: params.num_words_in_article]

    # Average representation of article (one vector for the whole article)
    if params.model_version == 'base' :
        # X_train_avg = np.zeros((len(X_train), 50)) # 50 b/c of 50D GloVe representation
        # for i in range(len(X_train)) :
        #     X_train_avg[i,:] = sentence_to_avg(X_train[i], word_to_vec_map)
        # print('..done averaging TRAIN set')

        # X_dev_avg = np.zeros((len(X_dev), 50)) # 50 b/c of 50D GloVe representation
        # for i in range(len(X_dev)) :
        #     X_dev_avg[i,:] = sentence_to_avg(X_dev[i], word_to_vec_map)
        # print('..done averaging DEV set')

        # X_test_avg = np.zeros((len(X_test), 50)) # 50 b/c of 50D GloVe representation
        # for i in range(len(X_test)) :
        #     X_test_avg[i,:] = sentence_to_avg(X_test[i], word_to_vec_map)
        # print('..done averaging TEST set')

        # np.savetxt('X_train_avg_data2.txt', X_train_avg, fmt='%f')
        # np.savetxt('X_dev_avg_data2.txt', X_dev_avg, fmt='%f')
        # np.savetxt('X_test_avg_data2.txt', X_test_avg, fmt='%f')
        X_train_avg = np.loadtxt('X_train_avg_data2.txt', dtype=float)
        X_dev_avg = np.loadtxt('X_dev_avg_data2.txt', dtype=float)
        X_test_avg = np.loadtxt('X_test_avg_data2.txt', dtype=float)


    # Building the model
    model = model_fn((params.num_words_in_article,), word_to_vec_map, word_to_index, params, nr_classes)
    # Printing out the model summary
    with open(os.path.join(args.model_dir, 'model_summary.txt'), 'w') as f:
        with redirect_stdout(f): model.summary()
    # Compiling the model
    opt = Adam(lr = params.learning_rate)
    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
    # Plotting the model
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    # Fitting/training the model (and setting the logger)
    csv_logger = CSVLogger(os.path.join(args.model_dir, 'training.log'))
    if params.model_version == 'base' :
        history = model.fit(X_train_avg, Y_train_oh, epochs = params.num_epochs, validation_data=(X_dev_avg, Y_dev_oh), batch_size = params.batch_size, shuffle=True, callbacks = [csv_logger])
    else :
        history = model.fit(X_train_indices, Y_train_oh, epochs = params.num_epochs, validation_data=(X_dev_indices, Y_dev_oh), batch_size = params.batch_size, shuffle=True, callbacks = [csv_logger])

    # Analysis and writing results to file
    # Train set loss/accuracy (all epochs)
    train_loss = history.history['loss']
    train_acc = history.history['acc']
    # Dev set loss/accuracy (all epochs)
    dev_loss = history.history['val_loss']
    dev_acc = history.history['val_acc']
    # Test set loss/accuracy (final epoch)
    if params.model_version == 'base' :
        test_loss, test_acc = model.evaluate(X_test_avg, Y_test_oh)
    else :
        test_loss, test_acc = model.evaluate(X_test_indices, Y_test_oh)
    # Predictions for train set
    if params.model_version == 'base' :
        Y_train_hat = model.predict(X_train_avg)
    else :
        Y_train_hat = model.predict(X_train_indices)
    Y_train_hat = np.argmax(Y_train_hat, axis = 1)
    # Predictions for dev set
    if params.model_version == 'base' :
        Y_dev_hat = model.predict(X_dev_avg)
    else :
        Y_dev_hat = model.predict(X_dev_indices)
    Y_dev_hat = np.argmax(Y_dev_hat, axis = 1)
    # Predictions for test set
    if params.model_version == 'base' :
        Y_test_hat = model.predict(X_test_avg)
    else :
        Y_test_hat = model.predict(X_test_indices)
    Y_test_hat = np.argmax(Y_test_hat, axis = 1)
    # Compute confusion matrix
    if nr_classes == 2 : class_names = ["Left", "Right"] # Labels - 0:Left, 1:Right
    elif nr_classes == 3 : class_names = ["Left", "Right", "Center"] # Labels - 0:Left, 1:Right, 2:Center
    cnf_matrix = confusion_matrix(Y_dev, Y_dev_hat)
    np.set_printoptions(precision=2)
    # Error Analysis
    train_all, train_misclass, dev_all, dev_misclass, test_all, test_misclass = error_analysis(train_dataset, Y_train_hat, dev_dataset, Y_dev_hat, test_dataset, Y_test_hat)
    # Elapsed time
    elapsed_time = time.time() - start_time
    print("Time: " + str(elapsed_time))
    # Writing to file
    file = open(os.path.join(args.model_dir, 'other_info.txt'), 'w')
    file.write('Dataset folder: ' + args.data_dir + '\n\n')
    file.write('Elapsed time [s]: ' + str(elapsed_time) + '\n\n')
    file.write('Train_loss: ' + str(train_loss[-1]) + '\n'), file.write('Dev_loss: ' + str(dev_loss[-1]) + '\n'), file.write('Test_loss: ' + str(test_loss) + '\n\n')
    file.write('Train_acc: ' + str(train_acc[-1]) + '\n'), file.write('Dev_acc: ' + str(dev_acc[-1]) + '\n'), file.write('Test_acc: ' + str(test_acc) + '\n\n')
    file.write('ERROR ANALYSIS :\n')
    file.write('TRAIN SET\n')
    file.write(str(train_misclass) + '\n\n')
    file.write('DEV SET\n')
    file.write(str(dev_misclass) + '\n\n')
    file.write('TEST SET\n')
    file.write(str(test_misclass) + '\n\n')
    file.close()
    # Save train_misclass_all to file
    train_all.to_csv(os.path.join(args.model_dir, 'misclass_train.csv'), index = False) # Could fix article word length
    dev_all.to_csv(os.path.join(args.model_dir, 'misclass_dev.csv'), index = False)
    test_all.to_csv(os.path.join(args.model_dir, 'misclass_test.csv'), index = False)
    # Saving final keras model
    model.save(os.path.join(args.model_dir, 'keras_model.h5'))

    # Plotting figures
    # Accuracy
    epochs = range(len(train_acc))
    fig1 = plt.figure()
    plt.plot(epochs, train_acc, '-', color = 'orange', label='training set')
    plt.plot(epochs, dev_acc, '-', color='blue', label='validation set')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    fig1.savefig(os.path.join(args.model_dir, 'accuracy.png'))
    # Loss
    fig2 = plt.figure()
    plt.plot(epochs, train_loss, '-', color='orange', label='training set')
    plt.plot(epochs, dev_loss,  '-', color='blue', label='validation set')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Cross-entropy loss')
    plt.legend()
    fig2.savefig(os.path.join(args.model_dir, 'loss.png'))
    # Confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes = class_names, title = 'Confusion matrix, without normalization')
    plt.savefig(os.path.join(args.model_dir, 'confusion.png'))
    # Confusion matrix (normalized)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes = class_names, normalize = True, title = 'Normalized confusion matrix')
    plt.savefig(os.path.join(args.model_dir, 'confusion_norm.png'))
    # ROC/AUC curves
    plot_roc(nr_classes, Y_dev, Y_dev_hat, args)



