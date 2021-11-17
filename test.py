import re
from matplotlib import pyplot as plt
import numpy as np
from numpy import linalg
# def read_file():
#     with open('t_input1.txt','r') as f:
#         for line in f:
#             line = line.split(' ')
#             print(line)
#             print(float(line[-1]))
# read_file()

def stat_grad_vector(A_new, b_new, x):
    h = 0.01
    ksi = np.random.random((6, len(x)))
    ksi = map(lambda k: k / (np.linalg.norm(k)), ksi)
    delta_grad = np.zeros(len(x))
    for k in ksi:
        k = np.array(k)
        delta = np.dot(A_new, x + h * k) - np.dot(A_new, x)
        delta_grad += delta * k
    return delta_grad / np.linalg.norm(delta_grad)


def stat_grad(f, x):
    h = 0.01
    ksi = np.random.uniform(0, 2, size=(6, len(x)))
    ksi = map(lambda k: k / (np.linalg.norm(k)), ksi)
    delta_grad = np.zeros(len(x))
    for k in ksi:
        k = np.array(k)
        delta = f(x + h * k) - f(x)
        delta_grad += delta * k
    return delta_grad / np.linalg.norm(delta_grad)


def func(x):
    return x[0] ** 2 - 4 * x[0] + x[1] ** 2 - 6 * x[1] + 13


x = np.array([1, 1], dtype=float)
x_prev = np.array([0, 0], dtype=float)
error = []
iter = 0
while (abs(func(x_prev) - func(x)) > 0.001 and iter < 10000):
    x_prev = x.copy()
    grad = np.array([2 * x[0] - 4, 2 * x[1] - 6])
    h = 0.1
    while func(x_prev) > func(h * grad):
        h *= 0.5
    x -= h * np.array([2 * x[0] - 4, 2 * x[1] - 6])
    error.append(linalg.norm(np.array([2, 3] - x)))
    iter += 1
error2 = []
iter = 0
x = np.array([1, 1], dtype=float)
x_prev = np.array([0, 0], dtype=float)
while (abs(func(x_prev) - func(x)) > 0.001 and iter < 10000):
    x_prev = x.copy()
    h = 0.1
    grad = stat_grad(func, x)
    while func(x_prev) > func(h * grad):
        h *= 0.5
    x -= h * grad
    error2.append(linalg.norm(np.array([2, 3] - x)))
    iter += 1
print(iter)
plt.plot(error)
plt.plot(error2, label='stat grad', color='red')
plt.show()
