import pandas as pd

if __name__ == '__main__':
    # load data
    data = pd.read_csv('data/kddcup.data.corrected')
    featureNameArray = pd.read_csv('data/kddcup.names.txt', header=None).values.ravel()

    data.columns = featureNameArray

    from preprocessing.KDD import myLabelEncoder
    encoder = myLabelEncoder()
    data = encoder.encode(data)

    from preprocessing.KDD import filter
    data = filter(data, 0.05)
    data = data.sample(frac = 0.5)
    X = data.iloc[:, 0:data.shape[1] - 1].values
    y = data.loc[:, "type"].values

    # construct test data 1, 2(split)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

    X_test_1, X_test_2, y_test_1, y_test_2 = train_test_split(X_test, y_test, test_size=.5)


    # construct test data 3
    test_data = pd.read_csv('data/corrected')
    test_data.columns = featureNameArray
    test_data = test_data.sample(frac=0.3)
    test_data = encoder.encode(test_data)
    X_test_3 = test_data.iloc[:, 0:test_data.shape[1] - 1].values
    y_test_3 = test_data.loc[:, "type"].values

    print("---------------")
    print("train : ", len(X_train))
    print("test 1 : ", len(X_test_1))
    print("test 2 : ", len(X_test_2))
    print("test 3 : ", len(X_test_3))

    # Conventional SVM
    from svm.conventionalSVM import ConventionalSVM
    csvm = ConventionalSVM()
    csvm.train(X_train, y_train)

    print("------Conventional Training Done-----")

    # Conventional test
    from printer import printResult

    csvm_predictions = csvm.test(X_test_1)
    printResult(y_test_1, csvm_predictions)

    csvm_predictions = csvm.test(X_test_2)
    printResult(y_test_2, csvm_predictions)

    csvm_predictions = csvm.test(X_test_3)
    printResult(y_test_3, csvm_predictions)

    # Enhanced SVM

