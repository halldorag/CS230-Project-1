

import numpy as np
import os
import matplotlib.pyplot as plt
import pdb

def create_plot(XData, YData, XLabel, BasePath):
	plt.ion()
	YLabel = 'Validation Accuracy'
	axes = plt.gca()
	axes.set_ylim([0.3, 0.9])
	plt.plot(XData, YData, 'bo')
	plt.ylabel(YLabel)
	plt.xlabel(XLabel)
	plt.show()


	plt.savefig(os.path.join(BasePath, (XLabel + 'png')))

def create_plot_log(XData, YData, XLabel, BasePath):
	plt.ion()
	YLabel = 'Validation Accuracy'
	axes = plt.gca()
	axes.set_ylim([0.3, 0.9])
	plt.plot(XData, YData, 'bo')
	plt.ylabel(YLabel)
	plt.xlabel(XLabel)
	plt.xscale("log")
	plt.show()

	plt.savefig(os.path.join(BasePath, (XLabel + 'png')))


# Learning Rate
LR = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
Val_Acc_LR = [0.342481703, 0.342481703, 0.769793746, 0.773952096, 0.740685296, 0.724717232]
Xlabel_LR = 'Learning Rate'

# Batch Size
BatchSize = [64, 128, 256, 512]
Val_Acc_Batch = [0.778775782, 0.780605455, 0.770292748, 0.749667333]
Xlabel_Batch = 'Batch Size'

SentenceLength = [100, 500, 1000, 2000, 3000]
Val_Acc_Sentence = [0.613772455, 0.801729874, 0.77578177, 0.820525615, 0.830671989]
Xlabel_Sentence = 'Sentence Length'


FilterSize = [64, 128, 256]
Val_Acc_Filter =[0.809880239, 0.830671989, 0.842814371]
Xlabel_Filter = 'Number of Filters'

BasePath = '/mnt/c/Users/Markus Zechner/Documents/GitHub/CS230-Project/temp/Simple_logistic_regression/data/model2/HyperParameters'

#create_plot_log(LR, Val_Acc_LR, Xlabel_LR, BasePath)

#create_plot(BatchSize, Val_Acc_Batch, Xlabel_Batch, BasePath)

#create_plot(SentenceLength, Val_Acc_Sentence, Xlabel_Sentence, BasePath)

create_plot(FilterSize, Val_Acc_Filter, Xlabel_Filter, BasePath)











