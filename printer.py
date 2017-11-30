def printResult(test_y, predictions):
    from sklearn.metrics import confusion_matrix, accuracy_score
    print("Precison : ", accuracy_score(test_y, predictions));
    cm = confusion_matrix(test_y, predictions)
    false_negative_rate = cm[1][0]/(cm[1][0]+cm[0][0])
    print("FNR : ", false_negative_rate)