import math
import matplotlib.pyplot as plt
import numpy as np
from sympy import *
import random
import copy


class LogReg:

    def __init__(self, data, learn_rate, eps):
        self.data_x = data[0]
        self.data_y = data[1]
        self.num_weights = len(self.data_x)
        self.num_data = len(self.data_y)
        self.weights = np.array([np.random.normal(0, 1) for _ in range(self.num_weights)])
        self.bias = np.random.normal(0, 1)
        self.learn_rate = learn_rate
        self.eps = eps

    def weight_sum(self):
        return self.bias + np.matmul(self.weights, self.data_x)

    def hypo(self):
        return 1 / (1 + np.exp(-1 * self.weight_sum()))

    def cost_func(self):
        # input to log10 might be close to 0 resulting in error -> try except block?
        term0 = np.dot(1 - self.data_y, np.log10(1 - self.hypo()))
        term1 = np.dot(self.data_y, np.log10(self.hypo()))
        cost = (-1/self.num_data) * (np.sum(term0) + np.sum(term1))
        return cost

    def gradient(self):
        diff = self.hypo() - self.data_y
        grad_b = (1/self.num_data) * (np.matmul(np.ones(self.num_data), np.array([diff]).T).T)[0]
        grad_w = (1/self.num_data) * (np.matmul(self.data_x, np.array([diff]).T).T)[0]
        return grad_b, grad_w

    def grad_desc(self):
        curr_cost = self.cost_func()
        prev_cost = 2 * curr_cost
        while (abs((curr_cost - prev_cost) / prev_cost) > self.eps):
            grad = self.gradient()
            self.bias -= self.learn_rate * grad[0]
            self.weights -= self.learn_rate * grad[1]
            prev_cost = curr_cost
            curr_cost = self.cost_func()

    def main(self):
        self.grad_desc()
        return self.bias, self.weights


class DataGen:

    def __init__(self, func, domain, size):
        self.func = func
        self.x_min = [x[0] for x in domain]
        self.x_max = [x[1] for x in domain]
        self.dim_v = size[0]
        self.dim_h = size[1]

    def get_org_data(self):
        data_x = np.array([[0 for _ in range(self.dim_h)] for _ in range(self.dim_v)])
        data_y = np.array([0 for _ in range(self.dim_h)])
        prev_zero = False
        for i in range(self.dim_h):
            satisfied = False
            while not satisfied:
                val_0 = np.random.randint(self.x_min[0], self.x_max[0])
                val_1 = np.random.randint(self.x_min[1], self.x_max[1])
                f_val = self.func.subs([('x0', val_0), ('x1', val_1)])
                if prev_zero and f_val >= 0:
                    satisfied = True
                    prev_zero = False
                    data_x[0][i] = val_0
                    data_x[1][i] = val_1
                    data_y[i] = 1
                elif not prev_zero and f_val < 0:
                    satisfied = True
                    prev_zero = True
                    data_x[0][i] = val_0
                    data_x[1][i] = val_1
                    data_y[i] = 0
        return (data_x, data_y)

    def get_data(self, data=()):
        if data == ():
            data_x, data_y = self.get_org_data()
            data_x, data_y = copy.deepcopy(data_x), copy.deepcopy(data_y) 
        else:
            data_x, data_y = copy.deepcopy(data[0]), copy.deepcopy(data[1])
        for i in range(self.dim_h):
            data_x[0][i] = data_x[0][i]**2        
        return (data_x, data_y)

    def get_data_outliers(self, perc, data=()):
        if data == ():
            data_x, data_y = self.get_data()
            data_x, data_y = copy.deepcopy(data_x), copy.deepcopy(data_y)
        else:
            data_x, data_y = copy.deepcopy(data[0]), copy.deepcopy(data[1])
        n = len(data_y)
        seen = set()
        for _ in range(int(perc * n)):
            i = random.randint(0, n-1)
            while i in seen:
                i = random.randint(0, n-1)
            seen.add(i)
            if data_y[i] == 0:
                data_y[i] = 1
            else:
                data_y[i] = 0
        return (data_x, data_y)


# SET-UP DATA

def get_domain(center, spans):
    return [(center[i] - spans[i] / 2, center[i] + spans[i] / 2) for i in range(len(center))]

def eval_func(func, input):
    new_func = func.subs([(f'x{i}', input[i]) for i in range(len(input))])
    return solve(new_func, f'x{len(input)}')

num_weights = 2
num_data = 20
size = (num_weights, num_data)
x0, x1 = symbols('x0 x1')
func = 1 - x0**2 + x1
spans = [10, 10]
input = [0]
val = eval_func(func, input)
domain = get_domain(input + val, spans)
out_perc = 0.1

datagen = DataGen(func, domain, size)
data_org_x, data_org_y = datagen.get_org_data()
data_x, data_y = datagen.get_data((data_org_x, data_org_y))
data_x_out, data_y_out = datagen.get_data_outliers(out_perc, (data_x, data_y))


# LOGISTIC REGRESSION

learn_rate = 0.005
eps = 10**(-6)

logreg = LogReg((data_x, data_y), learn_rate, eps)
bias, weights = logreg.main()
print(f'Bias: {bias}   Weights: {weights}')

logreg_out = LogReg((data_x_out, data_y_out), learn_rate, eps)
bias_out, weights_out = logreg_out.main()
print(f'Bias: {bias_out}   Weights: {weights_out}')


# PLOTTING

x_l = np.linspace(domain[0][0], domain[0][1], 100)

# data normal

y_l = [(-weights[0]*x**2 - bias) / weights[1] for x in x_l]

f1 = plt.figure()
for i in range(num_data):
    x0 = data_org_x[0][i]
    x1 = data_org_x[1][i]
    color = 'r'
    if data_org_y[i] == 0:
        color = 'b'
    plt.plot(x0, x1, f'{color}o')
plt.plot(x_l, y_l)

# data outliers

y_l_out = [(-weights_out[0]*x**2 - bias_out) / weights_out[1] for x in x_l]

f2 = plt.figure()
for i in range(num_data):
    x0 = data_org_x[0][i]
    x1 = data_org_x[1][i]
    color = 'r'
    if data_y_out[i] == 0:
        color = 'b'
    plt.plot(x0, x1, f'{color}o')
plt.plot(x_l, y_l_out)

plt.show()
