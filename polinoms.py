
from numpy import *


class polinom:
    def __init__(self):
        self.coef = []

    def cheb_value_in_point(self, power, point):
        if power == 0:
            return 1
        elif power == 1:
            return -1 + 2 * point
        else:
            return 2 * (-1 + 2 * point) * polinom.cheb_value_in_point(self, power-1, point) - polinom.cheb_value_in_point(self, power-2, point)

    def cheb_matrix_of_coefficient(self, max_power):
        matrix = zeros((max_power+1, max_power+1), float)
        matrix[0][0] = 1
        if max_power > 0:
            matrix[1][0] = -1
            matrix[1][1] = 2
        for i in range(2, max_power + 1):
            matrix[i][0] = -matrix[i-2][0] - 2 * matrix[i-1][0]
            for j in range(1, max_power + 1):
                matrix[i][j] = 4 * matrix[i-1][j-1] - matrix[i-2][j] - 2 * matrix[i-1][j]
        return matrix

    def Lag_value_in_point(self, power, point):
        if power == 0:
            return 1.0
        elif power == 1:
            return -point + 1
        else:
            return ((2.0 * (power - 1) + 1 - point) * polinom.Lag_value_in_point(self, power - 1, point) - (power - 1) * (power - 1)
                    * polinom.Lag_value_in_point(self, power - 2, point))

    def Lag_matrix_of_coefficient(self, max_power):
        matrix = zeros((max_power + 1, max_power + 1), float)
        matrix[0][0] = 1
        if max_power > 0:
            matrix[1][0] = 1
            matrix[1][1] = -1
        for i in range(2, max_power + 1):
            matrix[i][0] = -(i - 1) * (i - 1) * matrix[i - 2][0] + (2 * i - 1) * matrix[i - 1][0]
            for j in range(1, max_power + 1):
                matrix[i][j] = -matrix[i-1][j-1] - (i - 1) * (i - 1) * matrix[i-2][j] + (2 * i - 1) * matrix[i-1][j]
        return matrix

    def Ermit_value_in_point(self, power, point):
        if power == 0:
            return 1.0
        elif power == 1:
            return 2 * point
        else:
            return 2.0 * point * polinom.Ermit_value_in_point(self, power - 1, point) - 2 * (power - 1) * polinom.Ermit_value_in_point(self, power - 2, point)

    def Ermit_matrix_of_coefficient(self, max_power):
        matrix = zeros((max_power + 1, max_power + 1), float)
        matrix[0][0] = 1
        if max_power > 0:
            matrix[1][0] = 0
            matrix[1][1] = 2
        for i in range(2, max_power + 1):
            matrix[i][0] = -matrix[i - 2][0] * 2 * (i - 1)
            for j in range(1, max_power + 1):
                matrix[i][j] = 2 * matrix[i - 1][j - 1] - matrix[i - 2][j] * 2 * (i - 1)
        return matrix

    def Lejan_value_in_point(self, power, point):
        if power == 0:
            return 1.0
        elif power == 1:
            return point
        else:
            return ((2.0 * (power - 1) + 1) * point * polinom.Lejan_value_in_point(self, power - 1, point) - (power - 1)
                    * polinom.Lejan_value_in_point(self, power - 2, point)) / power

    def Lejan_matrix(self, max_power):
        matrix = zeros((max_power + 1, max_power + 1), float)
        matrix[0][0] = 1
        if max_power > 0:
            matrix[1][0] = 0
            matrix[1][1] = 1
        for i in range(2, max_power + 1):
            matrix[i][0] = -(i - 1) * matrix[i - 2][0] / i
            for j in range(1, max_power + 1):
                matrix[i][j] = ((2 * i - 1) * matrix[i - 1][j - 1] - (i-1) * matrix[i - 2][j]) / i
        return matrix

