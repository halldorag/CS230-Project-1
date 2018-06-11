
import os

path = '/mnt/c/Users/Markus Zechner/Documents/GitHub/CS230-Project/temp/Simple_logistic_regression/data/model/experiments/'
files = 0
folders = 0 
for _, dirnames, filenames in os.walk(path):
  # ^ this idiom means "we won't be using this value"
    files += len(filenames)
    folders += len(dirnames)


print("Found %d different folders to execute." % folders)


for i in range(folders):

    exe_command = "python3 main.py --model_dir experiments/exp_%s --data_dir data/dataset_1" % (i+1)
    #print(exe_command)
    os.system(exe_command)

