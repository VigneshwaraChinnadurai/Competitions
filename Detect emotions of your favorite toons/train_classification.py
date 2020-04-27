# organize imports
import shutil
import os
import pandas as pd
# get the conditions
pred = pd.read_csv('train.csv')

# get the input and output path
path  = r"C:\Users\91986\Desktop\Practice\Hacker Earth\Detect emotions of your favorite toons\Dataset\Train"

# get the class label limit
class_limit = 5

# variables to keep track
label = 0

# flower 102 class names
class_names = ["angry", "happy", "sad", "surprised","Unknown"]

# change the current working directory
os.chdir(path)

# creating the directories as required.
for x in range(1, class_limit+1):
	# create a folder for that class
	os.system("mkdir " + class_names[label])
	label += 1

# loop over the images in the dataset    
for n in pred['Frame_ID']:
    try:
        c=(pred[pred['Frame_ID']==n].iloc[0,1])
        shutil.move(n,path+"\\"+c+"\\"+n)
    except FileNotFoundError:
        pass