
import csv
import random


def generateDataWithLabelAndFeaturesFromCSVWithFirstColumnAsLabel (csv_filename) :
    with open (csv_filename) as f :
        reader = csv.reader(f)
        next(reader)

        data = []
        for row in reader :
            data.append({
                "label" : [int(cell) for cell in row[0]] ,
                "features" : [float(cell) for cell in row[0:] ]
            })



    return data

def testAndTrainSplit (data , ratio) :
    # Separate data into training and testing groups
    holdout = int(ratio * len(data))
    random.shuffle(data)
    testing = data[:holdout]
    training = data[holdout:]

    return testing , training


def filterOutIncorrectRows (data) :
    incorrect_rows = data[:1000]
    return incorrect_rows

def filterOutXandY (data) :
    X_data = [row["features"] for row in data]
    Y_data = [row["label"] for row in data]
    return X_data , Y_data

def calculateAccuracy (X_testing , Y_testing , data , predictions) :
    correct = 0
    incorrect = 0
    total = 0

    indexesOfIncorrect = []

    # for accuracy
    for actual , predicted in zip ( Y_testing , predictions ) :
        total += 1
        if actual == predicted :
            correct += 1
            indexesOfIncorrect.append(Y_testing.index(actual))
        else :
            incorrect += 1
            indexesOfIncorrect.append(Y_testing.index(actual))


    # for incorrect indexes
    incorrectXandYList = []
    for index in indexesOfIncorrect :
        incorrectXandYList.append( {
                "label" : Y_testing[index] ,
                "features" : X_testing[index]
            })

    indexListOfIncorrectItemsInData = []
    for incorrectItem in incorrectXandYList :
        indexListOfIncorrectItemsInData.append(data.index(incorrectItem))

    indexListOfIncorrectItemsInData = list(set(indexListOfIncorrectItemsInData))


    return correct , incorrect , indexListOfIncorrectItemsInData

def printResult (model , correct , incorrect ) :
    print(f"Result for model1 {type(model).__name__}")
    print(f"Correct : {correct}")
    print(f"InCorrect : {incorrect}")
    print(f"Accuracy : {100 * correct / (correct + incorrect)}%")
    print(f"\n \n ")
