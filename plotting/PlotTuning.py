
import numpy as np
import os
import matplotlib.pyplot as plt
import pdb

def create_plot(ParameterName, PlottingVariable, BasePath, NumberofExperiments, file_path, data, Legend):

	if PlottingVariable == 'acc':

		PlotColumn = 1
		YLabel = 'Training Accuracy'

	elif PlottingVariable == 'loss':

		PlotColumn = 2
		YLabel = 'Training Loss'

	elif PlottingVariable == 'val_acc':

		PlotColumn = 3
		YLabel = 'Validation Accuracy'

	elif PlottingVariable == 'val_loss':

		PlotColumn = 3
		YLabel = 'Validation Loss'

	XLabel = 'Epochs'


	for i in range(NumberofExperiments):

		plt.plot(data[i][:,0], data[i][:,PlotColumn])
		plt.legend(Legend)
		plt.ylabel(YLabel)
		plt.xlabel(XLabel)
		plt.show()

	plt.savefig(os.path.join(file_path, 'image.png'))



def plot_tuning(ParameterName, PlottingVariable, BasePath, NumberofExperiments, Legend):
    data = []

    file_path = os.path.join(BasePath, ParameterName)

    for i in range(NumberofExperiments):
		
        file_path_exp = os.path.join(file_path, 'exp_' + str(i+1))

        file_name = os.path.join(file_path_exp, 'training.log')

        data.append(np.genfromtxt(file_name, dtype=float, delimiter=',', skip_header=1))


    create_plot(ParameterName, PlottingVariable, BasePath, NumberofExperiments, file_path, data, Legend)
	






BasePath = '/mnt/c/Users/Markus Zechner/Documents/GitHub/CS230-Project/temp/Simple_logistic_regression/data/model2/HyperParameters'


Legend = ['0.1','0.01','0.001']
# Legend = ['64','128','256','512']
# Legend = ['100','1000','2000']
# Legend = ['64','128','256']
# Legend = ['0.25','0.5']

NumberofExperiments = len(Legend)

print(NumberofExperiments)

Parameters = ['LearningRate','BatchSize','SentenceLength', 'HiddenNodes']


pdb.set_trace()

# epoch,acc,loss,val_acc,val_loss



plot_tuning(Parameters[0], 'acc', BasePath, NumberofExperiments, Legend)

