import numpy as np
import scipy


class lin_model():
    def __init__(self, data, predictors, to_predict, constant = True) -> None:
        self.data = data
        self.predictors = predictors
        self.to_predict = to_predict
        self.constant = constant
        self.obs = len(data)
        self.X = None
        self.y = None
        self.estimates = None
        self.F = 0
        self.r_sq = 0
        self.resid = None
        self.RSS = 0
    
    def calc_estimates(self):
        self.X = np.ones((self.obs, 1))
        for predictor in self.predictors:
            self.X = np.column_stack((self.X, self.data[:, predictor]))
        self.y = self.data[:, self.to_predict]
        #OLS-estimator Beta = (X'X)^-1X'y
        self.estimates = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(self.X),self.X)), np.transpose(self.X)), self.y)
        self.resid = self.y - np.matmul(self.X, self.estimates)
        self.RSS = np.sum(np.power(self.resid, 2))
        self.calc_F()
        self.calc_r_sq()

    def calc_F(self):
        p = len(self.estimates)-1
        A = np.column_stack((np.zeros((p, 1)), np.eye(p)))
        Ab = A.dot(self.estimates.T)
        print(self.estimates)
        print(np.shape(Ab))
        S_sq = (1/(self.obs-len(self.estimates)))*self.RSS
        hat = np.matmul(np.transpose(self.X),self.X)
        print(np.shape(A))
        print(np.shape(hat))
        mid = np.linalg.inv(np.matmul(A, np.matmul(np.linalg.inv(hat), np.transpose(A))))
        self.F = np.transpose(Ab).dot(np.linalg.inv(np.matmul(A, np.matmul(np.linalg.inv(np.matmul(np.transpose(self.X),self.X)), np.transpose(A)))).dot(Ab))/(S_sq*(len(self.estimates)-1))
        print("Testi", self.F)

    def calc_r_sq(self):
        pass

    def fit_point(self, point):
        if len(self.estimates) != len(point):
            raise Exception("Invalid dimension of a point to be fitted")
        fitted = 0
        for i in range(point):
            fitted += point[i]*self.estimates[i]
    
    def p_val(self, statistic, value = None):
        if statistic == "F":
            return 1-scipy.stats.f.cdf(self.F, len(self.estimates)-1, (self.obs-len(self.estimates)))
    
    def print_vals(self):
        print("         ", "Estimate", "Std. Error", "t value", "p value")
        print("Intercept", self.estimates[0])
        i = 1
        for predictor in self.predictors:            
            print(predictor, "       ", self.estimates[i])
            i += 1
        print("F-statistic", self.F, ", p-value", self.p_val("F"))
        print("R-squared", self.r_sq)


credit = np.loadtxt("credit.csv", delimiter = ",")
model = lin_model(credit, [0,2,3], 1, True)
model.calc_estimates()
model.print_vals()