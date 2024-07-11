import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


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
    f1 = [2.31, 7.07, 7.07, 2.18, 2.18, 2.18, 7.87, 7.87, 7.87, 7.87]
    f2 = [65.2, 78.9, 61.1, 45.8, 54.2, 58.7, 96.1, 100.0, 85.9, 94.3]
    f3 = [15.3, 17.8, 17.8, 18.7, 18.7, 18.7, 15.2, 15.2, 15.2, 15.2]
    y = [24.0, 21.6, 34.7, 33.4, 36.2, 28.7, 27.1, 16.5, 18.9, 15.0]
    X, y = np.array([f1, f2, f3]).T, np.array(y).T

    model = CustomLinearRegression(fit_intercept=True)
    model.fit(X, y)
    yhat = model.predict(X)
    custom_dict = {'Intercept': model.intercept, 'Coefficient': model.coefficient,
                   'R2': model.r2_score(y, yhat), 'RMSE': model.rmse(y, yhat)}

    model_skl = LinearRegression(fit_intercept=True)
    model_skl.fit(X, y)
    yhat_skl = model_skl.predict(X)
    skl_dict = {'Intercept': float(model_skl.intercept_), 'Coefficient': model_skl.coef_,
                'R2': r2_score(y, yhat_skl), 'RMSE': float(np.sqrt(mean_squared_error(y, yhat_skl)))}

    print({x[0][0]: x[1][1] - x[0][1] for x in zip(custom_dict.items(), skl_dict.items())})


if __name__ == '__main__':
    main()
