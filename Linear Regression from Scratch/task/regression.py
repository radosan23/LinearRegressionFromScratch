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

    @staticmethod
    def r2_score(y, yhat):
        return float(1 - (sum((y - yhat) ** 2) / sum((y - np.mean(y)) ** 2)))

    @staticmethod
    def rmse(y, yhat):
        return float(np.sqrt(sum((y - yhat) ** 2) / y.size))


def main():
    capacity = [0.9, 0.5, 1.75, 2.0, 1.4, 1.5, 3.0, 1.1, 2.6, 1.9]
    age = [11, 11, 9, 8, 7, 7, 6, 5, 5, 4]
    cost_per_ton = [21.95, 27.18, 16.9, 15.37, 16.03, 18.15, 14.22, 18.72, 15.4, 14.69]
    X, y = np.array([capacity, age]).T, np.array(cost_per_ton).T

    model = CustomLinearRegression(fit_intercept=True)
    model.fit(X, y)
    yhat = model.predict(X)
    print({'Intercept': model.intercept, 'Coefficient': model.coefficient,
           'R2': model.r2_score(y, yhat), 'RMSE': model.rmse(y, yhat)})


if __name__ == '__main__':
    main()
