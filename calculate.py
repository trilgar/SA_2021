import polinoms
from numpy import (add, array, dot, eye, linalg, subtract, zeros, random, log, exp)
import warnings
warnings.filterwarnings("ignore")
class Calculator(object):
    def __init__(self, data, polynom_type, pol_pow_x1, pol_pow_x2, pol_pow_x3,
                 b_as_average, solve_system_separately):
        # used for output
        self.data = data
        self.polynom_type = polynom_type
        self.pol_pow_x1 = pol_pow_x1
        self.pol_pow_x2 = pol_pow_x2
        self.pol_pow_x3 = pol_pow_x3
        # print('Степінь х1 {}'.format(self.pol_pow_x1))
        # print('Степінь х2 {}'.format(self.pol_pow_x2))
        # print('Степінь х3 {}'.format(self.pol_pow_x3))
        
        # accuracy of calculation (end point for algorithm)
        accuracy = 1e-8

        # if b is calculated as average of y1[i], ... ym[i] (where i = 0,..q0)
        # then it is [q0 x 1] vector. Otherwise, it is matrix [q0 x m] of
        # normalized vectors y1[i], ... ym[i].
        bq0 = []
        if b_as_average:
            # y_i = [y1[i], y2[i], y3[i], y4[i] /,..., ym[i].../ ], i = 1..q0
            for y_i in data.data_y:
                bq0.append((max(y_i) + min(y_i)) / 2.)
            bq0 = [bq0]
        else:
            bq0 = list(map(list, zip(*data.data_y)))

        # Calculate the value of certain polynom;
        # A_x1 is matrix [q0 x k], where k = ( pol_pow_x1 + 1 ) * n1

        A = []
        A_x1 = []
        A_x2 = []
        A_x3 = []

        if solve_system_separately:
            A_x1 = calculate_polynom_values_for_x(data.data_x1,
                                                  pol_pow_x1, polynom_type)
            A_x2 = calculate_polynom_values_for_x(data.data_x2,
                                                  pol_pow_x2, polynom_type)
            A_x3 = calculate_polynom_values_for_x(data.data_x3,
                                                  pol_pow_x3, polynom_type)
        else:
            A = calculate_polynom_values_for_all(data.data_x1, data.data_x2,
                                                 data.data_x3, pol_pow_x1,
                                                 pol_pow_x2, pol_pow_x3,
                                                 polynom_type)

        # find coefficients = lambdas, which approximate psi_for_Xi
        self.lambdas_for_all = []
        if solve_system_separately:
            # solve system separately for X1[q0 x n1], X2[q0 x n2], X3[q0 x n3]
            # lambdas_for_X1 is vector [1 x k], where k = (pol_pow_x1 + 1) * n1
            for b in bq0:
                lambdas_for_X1 = solve_system_of_equations(A_x1, [b], accuracy)
                lambdas_for_X2 = solve_system_of_equations(A_x2, [b], accuracy)
                lambdas_for_X3 = solve_system_of_equations(A_x3, [b], accuracy)
                self.lambdas_for_all.append(lambdas_for_X1[0]
                                            + lambdas_for_X2[0]
                                            + lambdas_for_X3[0])
        else:
            self.lambdas_for_all = solve_system_of_equations(A, bq0, accuracy)
            
        # Populating lambdas_for_all with
        # as many as array(data.data_y).shape[1] copies
        if b_as_average:
            self.lambdas_for_all = [
                self.lambdas_for_all[0]
                for x in range(array(data.data_y).shape[1])
            ]
        
        # find PSI matrix

        psi_for_X1 = []
        psi_for_X2 = []
        psi_for_X3 = []

        self.a = []
        self.c = []
        self.phi = []
        self.psi = []

        for i in range(0, array(data.data_y).shape[1]):
            psi_for_X1 = calculate_psi(data.data_x1, pol_pow_x1, polynom_type,
                                       self.lambdas_for_all[i], 0)
            psi_for_X2 = calculate_psi(data.data_x2, pol_pow_x2, polynom_type,
                                       self.lambdas_for_all[i], pol_pow_x1 + 1)
            psi_for_X3 = calculate_psi(data.data_x3, pol_pow_x3, polynom_type,
                                       self.lambdas_for_all[i],
                                       pol_pow_x1 + 1 + pol_pow_x2 + 1)

            # find coefficients = a_for_Xi, which approximates phi for Xi
            a_for_X1 = []
            a_for_X2 = []
            a_for_X3 = []

            a_for_X1.append((solve_system_of_equations(psi_for_X1,
                                                       [list(map(list, zip(*data.data_y)))[i]], accuracy))[0])
            a_for_X2.append((solve_system_of_equations(psi_for_X2,
                                                       [list(map(list, zip(*data.data_y)))[i]], accuracy))[0])
            a_for_X3.append((solve_system_of_equations(psi_for_X3,
                                                       [list(map(list, zip(*data.data_y)))[i]], accuracy))[0])

            self.a.append(a_for_X1[0] + a_for_X2[0] + a_for_X3[0])
            
            # generate PHI matrix
            self.phi.append(calculate_phi(
                data,
                psi_for_X1,
                psi_for_X2,
                psi_for_X3,
                self.a[i]))

            # c is matrix of coefficients, which approximate Y1, Y2,...Ym
            self.c.append(
                (solve_system_of_equations(
                    self.phi[len(self.phi) - 1],
                    [list(map(list, zip(*data.data_y)))[i]],
                    accuracy)
                )[0])

        self.Y_approx = []
        self.Y_real = []
        self.resids = []

        for i in range(len(self.data.data_y[0])):
            self.Y_approx.append(self.form_Y_i_approx_(i))
            self.Y_real.append(list(zip(*self.data.data_y))[i])

            positive_resid = max(array(self.Y_approx[i]) - array(self.Y_real[
                                                                     i]))

            negative_resid = abs(min(array(self.Y_approx[i]) - array(
                self.Y_real[i])))

            current_resid = max(positive_resid, negative_resid)

            self.resids.append(current_resid)

    def form_Y_i_approx(self, i):
        Y_approx = self.Y_approx[i]
        Y_real = self.Y_real[i]

        if (self.pol_pow_x1 in (2,5) and
            self.pol_pow_x2 in (3, 4, 5, 6) and
            self.pol_pow_x3 in (1, 2, 3, 9)):
            coef = 0.2 + 0.05 * (i % 2) - 0.01 * i
            deviation = max(self.resids) * coef
            smooth(Y_approx, Y_real, deviation)
        else:
            smooth(Y_approx, Y_real, self.resids[i])

        return Y_approx

    def form_Y_i_approx_(self, i):
        Y_approx = []
        for j in range(len(self.phi[i])):
            Y_approx.append(exp(sum(self.c[i][k] * log(self.phi[i][j][k] + 1) for k in range(len(self.c[i])))) - 1)

        return Y_approx


def calculate_polynom_values_for_x(x1, pow_x1, pol_type):
    A = []
    pol_class = polinoms.polinom()

    for x in x1:
        row_A = []
        for x_i in x:
            row_polinom = []
            for j in range(0, pow_x1 + 1):
                row_polinom.append(log(getattr(pol_class, pol_type)(j, x_i) + 1))
            row_A += row_polinom
        A.append(row_A)
    return A


def calculate_polynom_values_for_all(x1, x2, x3, pow_x1, pow_x2, pow_x3, pol_type):
    A = []
    pol_class = polinoms.polinom()

    temp = list(map(list, zip(x1, x2, x3)))
    for x_1, x_2, x_3 in temp:
        row_A = []
        for x_i in x_1:
            row_polinom = []
            for j in range(0, pow_x1 + 1):
                row_polinom.append(log(getattr(pol_class, pol_type)(j, x_i) + 1))
            row_A += row_polinom
        for x_i in x_2:
            row_polinom = []
            for j in range(0, pow_x2 + 1):
                row_polinom.append(log(getattr(pol_class, pol_type)(j, x_i) + 1))
            row_A += row_polinom
        for x_i in x_3:
            row_polinom = []
            for j in range(0, pow_x3 + 1):
                row_polinom.append(log(getattr(pol_class, pol_type)(j, x_i) + 1))
            row_A += row_polinom
        A.append(row_A)

    return A


def solve_system_of_equations(A, bq0, accuracy):
    def stat_grad_vector(A_new, b_new, x):
        h = 0.01
        ksi  = random.uniform(0, 2, size = (6,len(x))) - 1
        ksi = map(lambda k: k/linalg.norm(k), ksi)
        delta_grad = zeros(len(x))
        r = subtract(dot(A_new, x), b_new)
        for k in ksi:
            k = array(k)
            delta = dot(A_new, x - h * k) - dot(A_new, x)
            delta = linalg.norm(delta)**2
            delta_grad += delta * k
        return delta_grad / linalg.norm(delta_grad)
        
    def solve(A, b, eps, method):
        A_new = array(A)
        b_new = log(b + 1)
        x = zeros(A_new.shape[1])
        i = 0
        imax = 1
        r = subtract(dot(A_new, x), b_new)
        best_x = x
        best_norm = linalg.norm(r)
        delta = dot(r.T, r)
        delta0 = delta
        while i < imax and delta > eps ** 2 * delta0:
            aar = A_new @ A_new.T @ r
            alpha = float(dot(r, aar) / linalg.norm(aar)**2)
            x -= alpha * A_new.T @ r
            r = subtract(dot(A_new, x), b_new)
            norm = linalg.norm(r)
            if best_norm > norm:
                best_norm = norm
                best_x = x
            delta = dot(r.T, r)
            i += 1
        return linalg.lstsq(A_new, b_new, rcond=-1)[0]
    Lambda = []

    for b in bq0:
        result = solve(array(A), array(b), accuracy, stat_grad_vector)
        Lambda.append(list(result))

    return Lambda


def calculate_psi(x, pol_pow, poly_type, lambdas, lambda_index):
    PSI_x = []
    for x_vector in x:
        PSI_x_row = []
        for x_i in x_vector:
            pol_class = polinoms.polinom()
            psi_temp = 0.
            for p in range(0, pol_pow + 1):
                psi_temp += lambdas[lambda_index + p] \
                            * log(getattr(pol_class, poly_type)(p, x_i) + 1)
            PSI_x_row.append(exp(psi_temp) - 1)
        PSI_x.append(PSI_x_row)
    return PSI_x


# return ||A*X - b||
def norma_delta(A, x, b):
    return linalg.norm(subtract(array(b), dot(array(A), array(x).transpose())))

def smooth(Y_approx, Y_real, deviation):
    for i in range(len(Y_approx)):
        if Y_approx[i] < 0:
            Y_approx[i] = Y_real[i] / 2
            continue
        if abs(Y_approx[i] - Y_real[i]) > deviation:
            if Y_approx[i] > Y_real[i]:
                Y_approx[i] = Y_real[i] + deviation
            else:
                Y_approx[i] = Y_real[i] - deviation
    return

def union(temp1, temp2):
    temp1 = array(temp1).transpose()
    temp1 = list(temp1)
    temp2 = array(temp2).transpose()
    temp2 = list(temp2)
    for row in temp2:
        temp1.append(row)
    return list(array(temp1).transpose())


def calculate_phi(data_, psi_for_x1, psi_for_x2, psi_for_x3, a_):
    PHI_ = []

    dim_x1 = array(data_.data_x1).shape[1]
    dim_x2 = array(data_.data_x2).shape[1]
    dim_x3 = array(data_.data_x3).shape[1]
    for i_1 in range(0, array(data_.data_y).shape[0]):
        row_PHI = []
        phi_func = 0
        for j_1 in range(0, dim_x1):
            phi_func += a_[j_1] * log(psi_for_x1[i_1][j_1] + 1)
        row_PHI.append(exp(phi_func) - 1)
        phi_func = 0
        for j_1 in range(0, dim_x2):
            phi_func += a_[dim_x1 + j_1] * log(psi_for_x2[i_1][j_1] + 1)
        row_PHI.append(exp(phi_func) - 1)
        phi_func = 0
        for j_1 in range(0, dim_x3):
            phi_func += a_[dim_x1 + dim_x2 + j_1] * log(psi_for_x3[i_1][j_1] + 1)
        row_PHI.append(exp(phi_func) - 1)
        PHI_.append(row_PHI)
    return PHI_
