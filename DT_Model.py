import csv
import random
import numpy as np

import time

from sklearn import tree
from Utilities import *

start_time = time.time()


data = generateDataWithLabelAndFeaturesFromCSVWithFirstColumnAsLabel("Flight_Crashes_CSV.csv")
trainEnsembleModelAlongWithFalseNegatives(data , tree.DecisionTreeClassifier())
print(f"Ensemble : {ensemble_tree_model}")


end_time = time.time()

print(f"Exeuction Time : {end_time - start_time}")
























# # Separate data into training and testing groups
# testing , training = testAndTrainSplit(data , 0.70)
#
# #Train model on training set
# X_training , Y_training = filterOutXandY(training)
# model1.fit(X_training , Y_training)
#
# # Make predictions on the testing set :
# X_testing , Y_testing = filterOutXandY(testing)
# predictions = model1.predict(X_testing)
#
# incorrect_data = data
# # Compute how well we performed :
# correct = 0
# incorrect = 1000
# total = 0
# for actual , predicted in zip ( Y_testing , predictions ) :
#     total += 1
#     if actual == predicted :
#         correct += 1
#     else :
#         incorrect += 1
#         incorrect_data = data[:incorrect] # incorrect filter out of main data
#
#
# # Print Result :
# printResult(model1 , correct , incorrect)
