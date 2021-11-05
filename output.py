import numpy as np
from copy import deepcopy
from numpy.polynomial import Polynomial as pm
from Compute import Compute

def basis_sh_chebyshev(degree):
    basis = [pm([-1, 2]), pm([1])]
    for i in range(degree):
        basis.append(pm([-2, 4])*basis[-1] - basis[-2])
    del basis[0]
    return basis

def basis_sh_legand(degree):
    basis = [pm([1])]
    for i in range(degree):
        if i == 0:
            basis.append(pm([-1, 2]))
            continue
        basis.append((pm([-2*i - 1, 4*i + 2])*basis[-1] - i * basis[-2]) / (i + 1))
    return basis

def basis_ermit(degree):
    basis = [pm([0]), pm([1])]
    for i in range(degree):
        basis.append(pm([0,2])*basis[-1] - 2 * i * basis[-2])
    del basis[0]
    return basis

def basis_lager(degree):
    basis = [pm([1])]
    for i in range(degree):
        if i == 0:
            basis.append(pm([1, -1]))
            continue
        basis.append(pm([2*i + 1, -1])*basis[-1] - i * i * basis[-2])
    return basis

class MyPolynom(object):
    #NOTE: eps was 1e-15
    def __init__(self, ar, symbol = 'x', eps = 1e-8):
        self.ar = ar
        self.polynom_symbol = symbol
        self.eps = eps
    
    def __repr__(self):
        #joinder[first, negative] = str
        joiner = {
            (True, True):'-',
            (True, False): '',
            (False, True): ' - ',
            (False, False): ' + '}
        
        result = []
        for deg, coef in reversed(list(enumerate(self.ar))):
            sign = joiner[not result, coef < 0]
            coef  = abs(coef)
            
            if coef == 1 and deg != 0:
                coef = ''
                
            if coef < self.eps:
                continue
                
            f = {0: '{}{}', 1: '{}{}' + self.polynom_symbol}.get(deg, '{}{}' + self.polynom_symbol + '^{}')
            result.append(f.format(sign, coef, deg))
            
        return ''.join(result) or '0'

class Output(object):
    def __init__(self, computation):
        self.computation = computation
        
        self.p = [self.computation.pol_pow_x1,
                  self.computation.pol_pow_x2,
                  self.computation.pol_pow_x3]
        
        self.deg = [np.array(self.computation.data.data_x1).shape[1],
                    np.array(self.computation.data.data_x2).shape[1],
                    np.array(self.computation.data.data_x3).shape[1],
                    np.array(self.computation.data.data_y).shape[1]]
        
        max_pow = max(self.p)
        
        if computation.polynom_type == "cheb_value_in_point":
            self.polynom_symbol = "T"
            self.basis = basis_sh_chebyshev(max_pow)
        
        elif computation.polynom_type == "Lejan_value_in_point":
            self.polynom_symbol = "P"
            self.basis = basis_sh_legand(max_pow)
        
        elif computation.polynom_type == "Lag_value_in_point":
            self.polynom_symbol = "L"
            self.basis = basis_lager(max_pow)
        
        elif computation.polynom_type == "Ermit_value_in_point":
            self.polynom_symbol = "H"
            self.basis = basis_ermit(max_pow)
        
        self.a = self.computation.a
        self.c = self.computation.c
        self.psi = self.computation.psi
        self.Lamb = np.matrix(self.computation.lambdas_for_all).transpose()
        
        self.minX = [[min(x) for x in self.computation.data.data_x1], 
                     [min(x) for x in self.computation.data.data_x2], 
                     [min(x) for x in self.computation.data.data_x3]]
                     
        self.maxX = [[max(x) for x in self.computation.data.data_x1], 
                     [max(x) for x in self.computation.data.data_x2], 
                     [max(x) for x in self.computation.data.data_x3]]
        
        self.minY = [min(x) for x in self.computation.data.data_y]
        self.maxY = [max(x) for x in self.computation.data.data_y]

    # Not done...
    def _form_lamb_lists(self):
        """
        Generates specific basis coefficients for Psi functions
        """
        self.psi = list()
        for i in range(np.array(self.computation.data.data_y).shape[1]):  # `i` is an index for Y
            psi_i = list()
            shift = 0
            for j in range(3):  # `j` is an index to choose vector from X
                psi_i_j = list()
                for k in range(self.deg[j]):  # `k` is an index for vector component
                    psi_i_jk = self.Lamb[shift:shift + self.p[j], i].getA1()
                    shift += self.p[j]
                    psi_i_j.append(psi_i_jk)
                psi_i.append(psi_i_j)
            self.psi.append(psi_i)

    # Done?
    def _transform_to_standard(self, _coeffs):
        """
        Transforms special polynomial to standard
        :param coeffs: coefficients of special polynomial
        :return: coefficients of standard polynomial
        """
        coeffs = np.array(_coeffs)
        std_coeffs = np.zeros(coeffs.shape)
        for index in range(coeffs.shape[0]):
            cp = self.basis[index].coef.copy()
            cp.resize(coeffs.shape)
            std_coeffs += coeffs[index] * cp
        return std_coeffs

    # Done?
    def _print_psi_i_jk(self, i, j, k):
        """
        Returns string of Psi function in special polynomial form
        :param i: an index for Y
        :param j: an index to choose vector from X
        :param k: an index for vector component
        :return: result string
        """
        strings = list()
        for n in range(len(self.psi[i][j][k])):
            strings.append("{0:.6f}*{symbol}{deg}(X{1}{2})".format(self.psi[i][j][k][n], j + 1, k + 1,
                                                                   symbol=self.polynom_symbol, deg=n))
        return ' + '.join(strings)

    # Done?
    def _print_phi_i_j(self, i, j):
        """
        Returns string of Phi function in special polynomial form
        :param i: an index for Y
        :param j: an index to choose vector from X
        :return: result string
        """

        strings = list()
        for k in range(len(self.psi[i][j])):
            shift = sum(self.deg[:j]) + k
            for n in range(len(self.psi[i][j][k])):
                strings.append('{0:.6f}*{symbol}{deg}(x{1}{2})'.format(self.a[i][shift] * self.psi[i][j][k][n],
                                                                       j + 1, k + 1, symbol=self.polynom_symbol, deg=n))
        return ' + '.join(strings)

    # Done?
    def _print_F_i(self, i):
        """
        Returns string of F function in special polynomial form
        :param i: an index for Y
        :return: result string
        """
        strings = list()
        for j in range(3):
            for k in range(len(self.psi[i][j])):
                shift = sum(self.deg[:j]) + k
                for n in range(len(self.psi[i][j][k])):
                    strings.append('{0:.6f}*{symbol}{deg}(x{1}{2})'.format(self.c[i][j] * self.a[i][shift] *
                                                                           self.psi[i][j][k][n],
                                                                           j + 1, k + 1, symbol=self.polynom_symbol, deg=n))
        return ' + '.join(strings)

    # Done?
    def _print_F_i_transformed_denormed(self, i):
        """
        Returns string of F function in special polynomial form
        :param i: an index for Y
        :return: result string
        """
        strings = list()
        constant = 0
        for j in range(3):
            for k in range(len(self.psi[i][j])):
                shift = sum(self.deg[:j]) + k
                raw_coeffs = self._transform_to_standard(self.c[i][j] * self.a[i][shift] * self.psi[i][j][k])
                diff = self.maxX[j][k] - self.minX[j][k]
                mult_poly = np.poly1d(np.array([1 / diff, - self.minX[j][k]]) / diff)
                add_poly = np.poly1d([1])
                current_poly = np.poly1d([0])
                for n in range(len(raw_coeffs)):
                    current_poly += add_poly * raw_coeffs[n]
                    add_poly *= mult_poly
                current_poly = current_poly * (self.maxY[i] - self.minY[i]) + self.minY[i]
                constant += current_poly[0]
                current_poly[0] = 0
                current_poly = np.poly1d(current_poly.coeffs, variable='(x{0}{1})'.format(j + 1, k + 1))
                strings.append(str(MyPolynom(current_poly, '(x{0}{1})'.format(j + 1, k + 1))))
        strings.append(str(constant))
        return ' +\n'.join(strings)

    # Done?
    def _print_F_i_transformed(self, i):
        """
        Returns string of F function in special polynomial form
        :param i: an index for Y
        :return: result string
        """
        strings = list()
        constant = 0
        for j in range(3):
            for k in range(len(self.psi[i][j])):
                shift = sum(self.deg[:j]) + k
                current_poly = np.poly1d(self._transform_to_standard(
                    self.c[i][j] * self.a[i][shift] * self.psi[i][j][k])[::-1],
                    variable = '(x{0}{1})'.format(j + 1, k + 1))
                constant += current_poly[0]
                current_poly[0] = 0
                strings.append(str(MyPolynom(current_poly, '(x{0}{1})'.format(j + 1, k + 1))))
        strings.append(str(constant))
        return ' +\n'.join(strings)

    # Done?
    def get_results(self):
        """
        Generates results based on given computation
        :return: Results string
        """
        self._form_lamb_lists()
        psi_strings = ['(Psi{1}{2})[{0}]={result}\n'.format(i + 1, j + 1, k + 1, result=self._print_psi_i_jk(i, j, k))
                       for i in range(np.array(self.computation.data.data_y).shape[1])
                       for j in range(3)
                       for k in range(self.deg[j])]
        phi_strings = ['(Phi{1})[{0}]={result}\n'.format(i + 1, j + 1, result=self._print_phi_i_j(i, j))
                       for i in range(np.array(self.computation.data.data_y).shape[1])
                       for j in range(3)]
        f_strings = ['(F{0})={result}\n'.format(i + 1, result=self._print_F_i(i))
                     for i in range(np.array(self.computation.data.data_y).shape[1])]
        f_strings_transformed_denormed = ['(F{0}) transformed ' \
                                          'denormed:\n{result}\n'.format(i + 1, result=self._print_F_i_transformed_denormed(i))
                                          for i in range(np.array(self.computation.data.data_y).shape[1])]
        f_strings_transformed = ['(F{0}) transformed:\n{result}\n'.format(i + 1, result=self._print_F_i_transformed(i))
                                 for i in range(np.array(self.computation.data.data_y).shape[1])]
                                 
        return '\n'.join(psi_strings + phi_strings + f_strings + f_strings_transformed_denormed + f_strings_transformed)

