import numpy as np
from numpy.polynomial import Polynomial as pm

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
    def __init__(self, ar, symbol = 'X', eps = 1e-8):
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
            deg = deg+1
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
        max_pow = max(self.computation.pol_pow_x1,
                      self.computation.pol_pow_x2,
                      self.computation.pol_pow_x3) + 1
        
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
            
        self.deg = [np.array(self.computation.data.data_x1).shape[1],
                    np.array(self.computation.data.data_x2).shape[1],
                    np.array(self.computation.data.data_x3).shape[1],
                    np.array(self.computation.data.data_y).shape[1]]
        
        self.minX = [np.array([min(x) for x in self.computation.data.data_x1]),
                     np.array([min(x) for x in self.computation.data.data_x2]), 
                     np.array([min(x) for x in self.computation.data.data_x3])]
                     
        self.maxX = [np.array([max(x) for x in self.computation.data.data_x1]), 
                     np.array([max(x) for x in self.computation.data.data_x2]), 
                     np.array([max(x) for x in self.computation.data.data_x3])]
        
        self.minY = np.array([min(x) for x in self.computation.data.data_y])
        self.maxY = np.array([max(x) for x in self.computation.data.data_y])
        
        self.a = np.array(self.computation.a)
        self.c = np.array(self.computation.c)
        
        self.psi = list()
        for i in range(self.deg[3]):
            shift = 0
            psi_i = list()
            psi_i_1 = list()
            for k in range(self.deg[0]):
                psi_i_1_k = list()
                for n in range(self.computation.pol_pow_x1 + 1):
                    psi_i_1_k.append(
                        self.computation.lambdas_for_all[i][shift + n]
                    )
                shift += self.computation.pol_pow_x1 + 1
                psi_i_1.append(np.array(psi_i_1_k))
            psi_i.append(psi_i_1)
                    
            psi_i_2 = list()
            for k in range(self.deg[1]):
                psi_i_2_k = list()
                for n in range(self.computation.pol_pow_x2 + 1):
                    psi_i_2_k.append(
                        self.computation.lambdas_for_all[i][shift + n]
                    )
                shift += self.computation.pol_pow_x2 + 1
                psi_i_2.append(np.array(psi_i_2_k))
            psi_i.append(psi_i_2)
                    
            psi_i_3 = list()
            for k in range(self.deg[2]):
                psi_i_3_k = list()
                for n in range(self.computation.pol_pow_x3 + 1):
                    psi_i_3_k.append(
                        self.computation.lambdas_for_all[i][shift + n]
                    )
                shift += self.computation.pol_pow_x3 + 1
                psi_i_3.append(np.array(psi_i_3_k))
            psi_i.append(psi_i_3)
            self.psi.append(psi_i)
    
    def _clever_join(self, key, strs):
        result = strs[0]
        for s in strs[1:]:
            if s.startswith('-') and key == ' + ':
                result += ' - ' + s[1:] 
            else:
                result += key + s
        return result
    
    def transform_to_standard(self, _coeffs):
        coeffs = np.array(_coeffs)
        std_coeffs = np.zeros(coeffs.shape)
        for index in range(coeffs.shape[0]):
            cp = self.basis[index].coef.copy()
            cp.resize(coeffs.shape)
            std_coeffs += coeffs[index] * cp
        return std_coeffs
   
    def form_F_i_polynomial(self, i):
        strings = list()
        constant = 0
        for j in range(3):
            for k in range(len(self.psi[i][j])):
                shift = sum(self.deg[:j]) + k
                current_poly = np.poly1d(self.transform_to_standard(
                    self.c[i][j] * self.a[i][shift] * self.psi[i][j][k])[::-1],
                    variable = '(X{0}{1})'.format(j + 1, k + 1))
                constant += current_poly[0]
                current_poly[0] = 0
                strings.append(str(MyPolynom(current_poly, '(X{0}{1})'.format(j + 1, k + 1))))
        strings.append(str(constant))
        return self._clever_join(' + ', strings)
        
    def form_F_i_polynomial_denormed(self, i):
        strings = list()
        constant = 0
        for j in range(3):
            for k in range(len(self.psi[i][j])):
                shift = sum(self.deg[:j]) + k
                raw_coeffs = self.transform_to_standard(self.c[i][j] * self.a[i][shift] * self.psi[i][j][k])
                diff = self.maxX[j][k] - self.minX[j][k]
                mult_poly = np.poly1d([1 / diff, - self.minX[j][k]] / diff)
                add_poly = np.poly1d([1])
                current_poly = np.poly1d([0])
                for n in range(len(raw_coeffs)):
                    current_poly += add_poly * raw_coeffs[n]
                    add_poly *= mult_poly
                current_poly = current_poly * (self.maxY[i] - self.minY[i]) + self.minY[i]
                constant += current_poly[0]
                current_poly[0] = 0
                current_poly = np.poly1d(current_poly.coeffs, variable='(X{0}{1})'.format(j + 1, k + 1))
                strings.append(str(MyPolynom(current_poly, '(X{0}{1})'.format(j + 1, k + 1))))
        strings.append(str(constant))
        return self._clever_join(' + ', strings)
    
    def form_result_string(self):
        result = ''
        
        result += 'Lambdas: \n'
        result += str(np.array(self.computation.lambdas_for_all)) + '\n'
        
        result += '\n\na: \n'
        for i in range(len(self.a)):
            result += 'Y' + str(i) + ': '
            result += str(self.a[i]) + '\n'
        
        
        result += '\n\nc: \n'
        for i in range(len(self.a)):
            result += 'Y' + str(i) + ': '
            result += str(self.c[i]) + '\n'
        
            
        result += '\n\n' 
        for i in range(len(self.c)):
            result += 'F' + str(i + 1) + '(' + ', '.join([
                'X' + str(j+1) for j in range(
                    len(self.c[i]))
            ]) + ') = '
            
            result += self._clever_join(' + ', [
                str(self.c[i][j]) + 
                ' * ' +
                'F' + str(i + 1) + str(j + 1) + 
                '(X' + str(j + 1) + ')'
                for j in range(len(self.c[i]))
            ])
                
            result += '\n'
        
        
        result += '\n\n'
        for i in range(len(self.a)):
            result += 'F' + str(i + 1) + '1' + '(X1) = '
            shift = 0
            result += self._clever_join(' + ', [
                str(self.a[i][shift + j]) + 
                ' * ' + 
                'PSI1' + str(j + 1) + 
                '[' + str(i + 1) + ']' +
                '(X1' + str(j + 1) + ')'
                for j in range(self.deg[0])
            ])
            
            result += '\n'
            
            result += 'F' + str(i + 1) + '2' + '(X2) = '
            shift += self.deg[0]
            result += self._clever_join(' + ', [
                str(self.a[i][shift + j]) + 
                ' * ' + 
                'PSI2' + str(j + 1) + 
                '[' + str(i + 1) + ']' +
                '(X2' + str(j + 1) + ')'
                for j in range(self.deg[1])
            ])
            
            result += '\n'
            
            result += 'F' + str(i + 1) + '3' + '(X3) = '
            shift += self.deg[1]
            result += self._clever_join(' + ', [
                str(self.a[i][shift + j]) + 
                ' * ' + 
                'PSI3' + str(j + 1) + 
                '[' + str(i + 1) + ']' +
                '(X3' + str(j + 1) + ')'
                for j in range(self.deg[2])
            ])
            
            result += '\n\n'
            
        result += '\n'
        
        for l in range(len(self.computation.lambdas_for_all)):
            shift = 0
            for i in range(self.deg[0]):
                result += 'PSI1' + str(i + 1) + '[' + str(l + 1) + '] = '
                result += self._clever_join(' + ', [
                        str(self.computation.lambdas_for_all[l][shift + j]) + 
                        ' * ' + 
                        self.polynom_symbol + str(j) +
                        '(X1' + str(i + 1) + ')'
                        for j in range(self.computation.pol_pow_x1 + 1)
                    ])
                shift += self.computation.pol_pow_x1 + 1
                result += '\n'
            
            for i in range(self.deg[1]):
                result += 'PSI2' + str(i + 1) + '[' + str(l + 1) + '] = '
                result += self._clever_join(' + ', [
                        str(self.computation.lambdas_for_all[l][shift + j]) + 
                        ' * ' + 
                        self.polynom_symbol + str(j) +
                        '(X2' + str(i + 1) + ')'
                        for j in range(self.computation.pol_pow_x2 + 1)
                    ])
                shift += self.computation.pol_pow_x2 + 1
                result += '\n'
                
            for i in range(self.deg[2]):
                result += 'PSI3' + str(i + 1) + '[' + str(l + 1) + '] = '
                result += self._clever_join(' + ', [
                        str(self.computation.lambdas_for_all[l][shift + j]) + 
                        ' * ' + 
                        self.polynom_symbol + str(j) +
                        '(X3' + str(i + 1) + ')'
                        for j in range(self.computation.pol_pow_x3 + 1)
                    ])
                shift += self.computation.pol_pow_x3 + 1
                result += '\n'
            
            result += '\n'
        
        result += '\nFi in polynomial form:\n'
        result += '\n'.join([
            'F' + str(i + 1) + ' = ' + 
            self.form_F_i_polynomial(i)
            for i in range(self.deg[3])
        ])
        
        result += '\n\n\nFi in polynomial denormed form:\n'
        result += '\n'.join([
            'F' + str(i + 1) + ' = ' +
            self.form_F_i_polynomial_denormed(i)
            for i in range(self.deg[3])
        ])
        
        return result
