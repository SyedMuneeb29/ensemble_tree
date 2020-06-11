import csv
import random
import numpy as np


from sklearn import tree
from Utilities import *


ensemble_tree_model = [

]

def appendEnsembleTreeModel (model , false_negative_list) :
    ensemble_tree_model.append({
        "model"+str(len(ensemble_tree_model)) : model ,
        "false_negative_list"+str(len(ensemble_tree_model)) : false_negative_list
    })

def trainEnsembleModelAlongWithFalseNegatives (data , ml_model) :

    # separate training and testing data
    testing , training = testAndTrainSplit(data , 0.90)

    # x and y from training and testing
    X_training , Y_training = filterOutXandY(training)
    X_testing , Y_testing = filterOutXandY(testing)

    # train

    try :

        ml_model.fit(X_training , Y_training)

        # predict
        predictions = ml_model.predict(X_testing)

        # accuracy of the model :
        correct , incorrect , indexListOfIncorrectInData = calculateAccuracy(X_testing , Y_testing , data , predictions)

        printResult(ml_model , correct , incorrect)

        # if incorrect != 0 :
        for i in range(1) :
            false_negative_list = []

            for index in indexListOfIncorrectInData :
                false_negative_list.append(data[index])

            appendEnsembleTreeModel(type(ml_model).__name__ , indexListOfIncorrectInData)

            # print(false_negative_list)
            newDataForNextModel = false_negative_list

            trainEnsembleModelAlongWithFalseNegatives(newDataForNextModel , tree.DecisionTreeClassifier())

    except :
        print("Model Learned")




########  MODEL 1 ##########

model1 = tree.DecisionTreeClassifier()


data = generateDataWithLabelAndFeaturesFromCSVWithFirstColumnAsLabel("MQ2008_CSV.csv")



trainEnsembleModelAlongWithFalseNegatives(data , tree.DecisionTreeClassifier())


print(f"Ensemble : {ensemble_tree_model}")

# trainEnsembleModelAlongWithFalseNegatives(filterOutIncorrectRows(data) , tree.DecisionTreeClassifier())




############## MODEL 2 ####################






















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
