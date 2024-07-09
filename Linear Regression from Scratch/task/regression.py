import numpy as np


class CustomLinearRegression:
    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficient = None
        self.intercept = None

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        coefs = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
        if self.fit_intercept:
            self.intercept = float(coefs[0])
            self.coefficient = coefs[1:]
        else:
            self.coefficient = coefs

    def predict(self, X):
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
            return np.dot(X, np.hstack((self.intercept, self.coefficient)))
        return np.dot(X, self.coefficient)


def main():
    x = [4, 4.5, 5, 5.5, 6, 6.5, 7]
    w = [1, -3, 2, 5, 0, 3, 6]
    z = [11, 15, 12, 9, 18, 13, 16]
    y = [33, 42, 45, 51, 53, 61, 62]
    X, y = np.array([x, w, z]).T, np.array(y).T

    model = CustomLinearRegression(fit_intercept=False)
    model.fit(X, y)
    print(model.predict(X))


if __name__ == '__main__':
    main()
