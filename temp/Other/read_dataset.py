import csv
import random
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
font = {'family' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)
from matplotlib import pyplot as plt
def read_and_split_dataset(filename, small_or_all):
	"""
	This function reads in the Quora dataset. It outputs 12
	files. 4 files for training, 4 for development, and 4 for
	testing. 70% of the data goes to training, 15% goes to 
	development, and 15% goes to testing.
	"""
	if small_or_all == 1:
		directory = "split_data_small"
	else:
		directory = "split_data"
	# open files
	train_q1    = open("{}/train.question1.txt".format(directory), "w")
	train_q2    = open("{}/train.question2.txt".format(directory), "w")
	train_label = open("{}/train.label.txt".format(directory), "w")
	train_id    = open("{}/train.id.txt".format(directory), "w")

	val_q1    = open("{}/val.question1.txt".format(directory), "w")
	val_q2    = open("{}/val.question2.txt".format(directory), "w")
	val_label = open("{}/val.label.txt".format(directory), "w")
	val_id    = open("{}/val.id.txt".format(directory), "w")

	test_q1    = open("{}/test.question1.txt".format(directory), "w")
	test_q2    = open("{}/test.question2.txt".format(directory), "w")
	test_label = open("{}/test.label.txt".format(directory), "w")
	test_id    = open("{}/test.id.txt".format(directory), "w")
	length = []
	with open(filename) as tsvfile:
		tsvreader = csv.reader(tsvfile, delimiter="\t")
		counter = 0
		for line in tsvreader:
			# read in each line and create the questions dict
			if line[3] and line[4] and line[5] and line[0]:
			# uncomment below two lines for a small dataset
				if (small_or_all == 1) and (counter > 10000):
					break
				if counter > 0:
					random_num = random.random()
					if random_num <= 0.7:
						length.append(len(line[3].replace('\n',' ')))
						train_q1.write("{}\n".format(separate_words_from_symbols(line[3].replace('\n',' '))))
						train_q2.write("{}\n".format(separate_words_from_symbols(line[4].replace('\n',' '))))
						train_label.write("{}\n".format(line[5].replace('\n',' ')))
						train_id.write("{}\n".format(line[0]))
					elif (random_num > 0.7) and (random_num <= 0.85):
						val_q1.write("{}\n".format(separate_words_from_symbols(line[3].replace('\n',' '))))
						val_q2.write("{}\n".format(separate_words_from_symbols(line[4].replace('\n',' '))))
						val_label.write("{}\n".format(line[5].replace('\n',' ')))
						val_id.write("{}\n".format(line[0].replace('\n',' '))) 
					else:
						test_q1.write("{}\n".format(separate_words_from_symbols(line[3].replace('\n',' '))))
						test_q2.write("{}\n".format(separate_words_from_symbols(line[4].replace('\n',' '))))
						test_label.write("{}\n".format(line[5].replace('\n',' ')))
						test_id.write("{}\n".format(line[0].replace('\n',' ')))
				counter += 1
		bins = np.arange(0, 200, 10) 
		print(np.mean(np.asarray(length)))
		print(np.std(np.asarray(length)))
		plt.hist(np.asarray(length),bins=bins,color = "#C00000")
		plt.xlabel('Word length')
		plt.ylabel('Occurrences')
		plt
		plt.savefig("pp.pdf", format='pdf')
		plt.close("all")
		objects = ('Non-Duplicate', 'Duplicate')
		y_pos = np.arange(len(objects))
		number = [178357,104803]
		plt.bar(y_pos, number, align='center', color ="#C00000",width = 0.3)
		plt.xticks(y_pos, objects)
		plt.ylabel('Occurrences')
		plt.savefig("bar.pdf", format='pdf')

		
		


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
	#question = re.sub('\"', ' " ', question)
	question = re.sub("\/", " / ", question)
	question = re.sub("\:", " : ", question)
	# question = re.sub("[\\]", " \ ", question)
	return question

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print "Usage:\n"
		print "$ python read_dataset.py [1 for small|2 for all]"
		print "e.g. python read_dataset.py 1"
		sys.exit(0)
	small_or_all = int(sys.argv[1])
	filename = "quora_duplicate_questions.tsv"
	read_and_split_dataset(filename, small_or_all)