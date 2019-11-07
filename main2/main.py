###SECOND MAIN


"""
This is a sample script for PRMU Algorithm Contest 2019.
You can modify the codes anywhere. 
Make sure the output format.
"""

import numpy as np
import pandas
import time
from srcs.utils import AlconDataset, AlconTargets, unicode_to_kana_list, evaluation
from srcs.myalgorithm import MyAlgorithm

""" 
Main Script 
"""
# Set dataset path manually
datapath='./../dataset/';

# Load dataset
targets = AlconTargets(datapath, train_ratio=1)
# print(len(targets.train))     // Train part from training set
# print(len(targets.val))       // Test part from the training set
# print(len(targets.test))      // Test set
traindata = AlconDataset(datapath, targets.train, isTrainVal=True)
valdata = AlconDataset(datapath,targets.val, isTrainVal=True)
testdata = AlconDataset(datapath,targets.test, isTrainVal=False)

# Train model
print('Building model ...', end='')
myalg = MyAlgorithm()  # Your algorithm class
myalg.build_model( traindata )  # Train my model

print('done'+str(len(traindata)))

# Prediction
N = len(valdata)
sheet = valdata.getSheet()  # Get initial sheet

start = time.time()

for i in range(N):
    break
    # Prediction
    img,y_val = valdata[i]  # Get data
    # print("aa")
    y_pred = myalg.predict(img)  # Prediction
    # print("ss")

    if(i%100 == 0):
        end = time.time()
        print('Prediction {}; GT {}; {}/{}'.format(
            unicode_to_kana_list(y_pred),unicode_to_kana_list(y_val),i,N))
        print("Take " , end-start)
        start = time.time()

    # Fill the sheet with y_pred
    sheet.iloc[i,1:4] = y_pred

# Evaluation
acc = evaluation( sheet, valdata.targets )
print('Accuracy = {:f} (%)'.format(acc))

# save predicted results in CSV
# Zip and submit it.
sheet.to_csv('test_prediction.csv',index=False)  
