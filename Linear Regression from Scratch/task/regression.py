import numpy as np


class CustomLinearRegression:
    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficient = None
        self.intercept = None

    def fit(self, X, y):
        X, y = np.array(X).reshape(-1, 1), np.array(y)
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        coefs = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
        if self.fit_intercept:
            self.intercept = float(coefs[0])
            self.coefficient = coefs[1:]
        else:
            self.coefficient = coefs


def main():
    x = [4.0, 4.5, 5, 5.5, 6.0, 6.5, 7.0]
    y = [33, 42, 45, 51, 53, 61, 62]

    model = CustomLinearRegression()
    model.fit(x, y)
    print({'Intercept': model.intercept, 'Coefficient': model.coefficient})


if __name__ == '__main__':
    main()
