
import csv
import random

ensemble_tree_model = [

]

def appendEnsembleTreeModel (model , false_negative_list) :
    ensemble_tree_model.append({
        "model"+str(len(ensemble_tree_model)) : model ,
        "false_negative_list"+str(len(ensemble_tree_model)) : false_negative_list
    })

def trainEnsembleModelAlongWithFalseNegatives (data , ml_model) :

    # separate training and testing data
    testing , training = testAndTrainSplit(data , 0.70)

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
