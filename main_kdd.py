import pandas as pd
import numpy as np

sample_number = 300000

if __name__ == '__main__':
    # load data
    data = pd.read_csv('data/kddcup.data_10_percent_corrected')
    featureNameArray = pd.read_csv('data/kddcup.names.txt', header=None).values.ravel()

    data.columns = featureNameArray

    from preprocessing.KDD import myLabelEncoder
    encoder = myLabelEncoder()
    data = encoder.encode(data)

    from preprocessing.KDD import filter
    data = filter(data, 0.015)
    data = data.sample(300000)
    X = data.iloc[:, 0:data.shape[1] - 1].values
    y = data.loc[:, "type"].values

    # construct test data 1, 2(split)
    from sklearn.model_selection import train_test_split
    X_train, X_test_data, y_train, y_test_data = train_test_split(X, y, test_size=100000)

    X_test_1, X_test_2, y_test_1, y_test_2 = train_test_split(X_test_data, y_test_data, test_size=50000)
    X_test_list = [X_test_1,X_test_2]
    y_test_list = [y_test_1, y_test_2];

    # construct test data 3
    test_data = pd.read_csv('data/corrected')
    test_data.columns = featureNameArray
    test_data = test_data.sample(10000)
    test_data = encoder.encode(test_data)
    temp_x = test_data.iloc[:, 0:test_data.shape[1] - 1].values
    X_test_list.append(temp_x);
    y_test_list.append(test_data.loc[:, "type"].values);

    print("---------------")
    print("train : ", len(X_train))
    for index in range(0, len(X_test_list)) :
        print("test %d : %s" % (index, len(X_test_list[index])))

    # Conventional SVM
    print("------Conventional Training Start-----")
    from svm.conventionalSVM import ConventionalSVM
    csvm = ConventionalSVM()
    csvm.train(X_train, y_train)

    print("------Conventional Training Done-----")

    # Conventional test
    from printer import printResult

    for index in range(0, len(X_test_list)):
        csvm_predictions = csvm.test(X_test_list[index])
        printResult(y_test_list[index], csvm_predictions)

    print("------Conventional Test Done-----")

    # Enhanced SVM
    print("------Enhanced Training Start-----")
    from preprocessing.RoughSet import RoughSet
    RoughSet().getReducts(data);