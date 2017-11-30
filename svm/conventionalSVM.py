class ConventionalSVM():
    def train(self, X, y):
        # train SVM
        from sklearn import svm
        self.clf = svm.SVC(kernel="rbf", gamma=pow(10, -6))
        self.clf.fit(X, y)

    def test(self, test_X):
        return self.clf.predict(test_X)
