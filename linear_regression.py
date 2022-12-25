import numpy as np
import scipy.stats
import math


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
        self.S_sq = 0
        self.resid = None
        self.RSS = 0
        self.errors = np.zeros((len(self.predictors)+1))
        self.t = None
    
    def calc_estimates(self):
        self.X = np.ones((self.obs, 1))
        for predictor in self.predictors:
            self.X = np.column_stack((self.X, self.data[:, predictor]))
        self.y = self.data[:, self.to_predict]
        #OLS-estimator Beta = (X'X)^-1X'y
        self.estimates = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(self.X),self.X)), np.transpose(self.X)), self.y)
        self.resid = self.y - np.matmul(self.X, self.estimates)
        self.RSS = np.sum(np.power(self.resid, 2))
        hat = np.linalg.inv(np.matmul(np.transpose(self.X), self.X))
        self.S_sq = (1/(self.obs-len(self.estimates)))*self.RSS
        for i in range(len(self.estimates)):
            self.errors[i] = math.sqrt(self.S_sq * hat[(i,i)])
        self.t = self.estimates/self.errors
        self.calc_F()
        self.calc_r_sq()

    def calc_F(self):
        p = len(self.estimates)-1
        A = np.column_stack((np.zeros((p, 1)), np.eye(p)))
        Ab = A.dot(self.estimates.T)
        hat = np.matmul(np.transpose(self.X),self.X)
        mid = np.linalg.inv(np.matmul(A, np.matmul(np.linalg.inv(hat), np.transpose(A))))
        self.F = np.transpose(Ab).dot(np.linalg.inv(np.matmul(A, np.matmul(np.linalg.inv(np.matmul(np.transpose(self.X),self.X)), np.transpose(A)))).dot(Ab))/(self.S_sq*(len(self.estimates)-1))
    def calc_r_sq(self):
        avg = np.average(self.y)
        TSS = 0
        for y in self.y:
            TSS += (y-avg)**2
        self.r_sq = 1- self.RSS/TSS
    

    def fit_point(self, point):
        if len(self.estimates) != len(point):
            raise Exception("Invalid dimension of a point to be fitted")
        fitted = 0
        for i in range(point):
            fitted += point[i]*self.estimates[i]
    
    def p_val(self, statistic, value = None):
        if statistic == "F":
            return 1-scipy.stats.f.cdf(self.F, len(self.estimates)-1, (self.obs-len(self.estimates)))
        elif statistic == "t":
            return (1- scipy.stats.t.cdf(abs(value), self.obs-len(self.estimates)))*2
    
    def print_vals(self):
        print("         ", "Estimate", "Std. Error", "t value", "p value")
        print("Intercept", round(self.estimates[0],2), round(self.errors[0],2), round(self.t[0], 2),"    ", round(self.p_val("t",self.t[0]), 2))
        i = 1
        for predictor in self.predictors:            
            print(predictor, "       ", round(self.estimates[i],2), round(self.errors[i], 2), round(self.t[i],2),"    ", round(self.p_val("t",self.t[i]), 6))
            i += 1
        print("F-statistic", self.F, ", p-value", self.p_val("F"))
        print("R-squared", self.r_sq)


#credit = np.loadtxt("credit.csv", delimiter = ",")
#model = lin_model(credit, [0,2,3], 1, True)
#model.calc_estimates()
#model.print_vals()