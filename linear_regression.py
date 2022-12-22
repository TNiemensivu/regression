import numpy as np


class lin_model():
    def __init__(self, data, predictors, to_predict, constant = True) -> None:
        self.data = data
        self.predictors = predictors
        self.to_predict = to_predict
        self.constant = constant
        self.obs = len(data)
        self.estimates = None
    
    def calc_estimates(self):
        X = np.ones((self.obs, 1))
        for predictor in self.predictors:
            X = np.column_stack((X, self.data[:, predictor]))
        y = self.data[:, self.to_predict]
        #OLS-estimator Beta = (X'X)^-1X'y
        self.estimates = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)), np.transpose(X)), y)
    
    def fit_point(self, point):
        if len(self.estimates) != len(point):
            raise Exception("Invalid dimension of a point to be fitted")
        fitted = 0
        for i in range(point):
            fitted += point[i]*self.estimates[i]
    
    def print_vals(self):
        print("         ", "Estimate", "Std. Error", "t value", "p value")
        print("Intercept", self.estimates[0])
        i = 1
        for predictor in self.predictors:            
            print(predictor, "       ", self.estimates[i])
            i += 1



credit = np.loadtxt("credit.csv", delimiter = ",")
model = lin_model(credit, [0,2,3], 1, True)
model.calc_estimates()
model.print_vals()