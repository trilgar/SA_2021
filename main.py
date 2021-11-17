import pprint
import sys
import os
import matplotlib
import numpy as np
matplotlib.use('Qt5Agg')
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from matplotlib import pyplot as plt
from PyQt5.uic import loadUi
from calculate import Calculator
from output_shape import Output
from data_loader import DataLoader
from multiprocessing import Pool

class App(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = loadUi('main.ui')
        self.ui.input_file_button.clicked.connect(self.openInputFileClicked)
        self.ui.output_file_button.clicked.connect(self.openOutFileClicked)
        self.ui.calculate.clicked.connect(self.compute)
        self.ui.plot.clicked.connect(self.plotGraphic)
    def openInputFileClicked(self):
        filename = QFileDialog.getOpenFileName(self, "Вхідний файл:", "")
        filename = str(filename[0]).split('/')[-1]
        print(filename)
        print(type(filename))
        self.ui.input_file.setText(filename)
    
    def openOutFileClicked(self):
        filename = QFileDialog.getOpenFileName(self, "Файл виводу", "")
        filename = str(filename[0]).split('/')[-1]
        self.ui.output_file.setText(filename)
        
       
    
    def compute(self):
        # read configuration data
        dim_x1 = int(self.ui.input_x1_size.text()) if self.ui.input_x1_size.text() != '' else 2 
        dim_x2 = int(self.ui.input_x2_size.text()) if self.ui.input_x2_size.text() != '' else 2 
        dim_x3 = int(self.ui.input_x3_size.text()) if self.ui.input_x3_size.text() != '' else 3 
        dim_y = int(self.ui.input_y_size.text()) if self.ui.input_y_size.text() != '' else 4
        sample_range = int(self.ui.sample_size.text()) if self.ui.sample_size.text()!= '' else 40

        input_file_name = self.ui.input_file.text() if self.ui.input_file.text() != '' else 'lab3.txt'
        self.output_file = self.ui.output_file.text() if self.ui.output_file.text()!='' else 'output.txt'

        grid_search = self.ui.grid_search.isChecked()
        if not grid_search:
            pol_pow_x1 = int(self.ui.x1_pow.text()) if self.ui.x1_pow.text() != '' else 1
            pol_pow_x2 = int(self.ui.x2_pow.text()) if self.ui.x2_pow.text() != '' else 1
            pol_pow_x3 = int(self.ui.x3_pow.text()) if self.ui.x3_pow.text() != '' else 1
        #grid_search = self.ui.grid_search.isChecked()
            
            
        bq0AsAvg = True
        bq0AsAvg = self.ui.max_min.isChecked()
        lambda_separate = True
        lambda_separate = self.ui.lambda_sep.isChecked()

        if self.ui.cheb.isChecked():
            polynom_type = "cheb_value_in_point"
        elif self.ui.lagger.isChecked():
            polynom_type = "Lag_value_in_point"
        elif self.ui.ermit.isChecked():
            polynom_type = "Ermit_value_in_point"
        elif self.ui.lejandr.isChecked():
            polynom_type = "Lejan_value_in_point"
        else:
            polynom_type = "cheb_value_in_point"

        # Fills dropping list of Yi for plotting
        self.ui.var_choice.clear()
        self.ui.var_choice.addItems(["Y" + str(i) for i in range(0, dim_y)])

        # read data from input
        self.data = DataLoader(input_file_name, dim_x1, dim_x2, dim_x3, dim_y, sample_range)
        if self.data.ERROR:
            exit(1)

        self.normalized_data = DataLoader(
            data_x1=self.data.data_x1,
            data_x2=self.data.data_x2,
            data_x3=self.data.data_x3,
            data_y=self.data.data_y)

        normalize_data_container(self.normalized_data)

        
        if grid_search:
            all_pos_pow = []
            for pow_x1 in range(0,5):
                for pow_x2 in range(0,5):
                    for pow_x3 in range(0,5):
                        all_pos_pow.append((pow_x1,pow_x2,pow_x3))
            resids = []
            args = []
            for item in all_pos_pow:
                args.append([self.normalized_data, polynom_type, item[0], item[1], item[2], bq0AsAvg,
                               lambda_separate])
            pool = Pool(processes=12)
            
            
            resids = pool.map(calculate,args)
            min_resid = 100
            for x1_pow,x2_pow,x3_pow, resid in resids:
                if(np.mean(resid)< min_resid):
                    min_resid = np.mean(resid)
                    pol_pow_x1 = x1_pow
                    pol_pow_x2 = x2_pow
                    pol_pow_x3 = x3_pow
            # print(best_res)
        print('pows:', pol_pow_x1, pol_pow_x2, pol_pow_x3)
        self.program = Calculator(self.normalized_data, polynom_type, pol_pow_x1, pol_pow_x2, pol_pow_x3, bq0AsAvg,
                                  lambda_separate)

        # output
        output = Output(self.program)
        #result_string = output.get_results()
        result_string = output.form_result_string()
        #self.ui.output.setText(result_string)
        
        with open(self.output_file, "w") as out_file:
            out_file.write(result_string)
        local_path = "output.txt"
        os.startfile(local_path)
            

    def plotGraphic(self):
        # Disabling toolbar
        matplotlib.rcParams['toolbar'] = 'None'
        plt.style.use('ggplot')

        i = self.ui.var_choice.currentIndex()

        Y_approx = self.program.form_Y_i_approx(i)
        Y_real = list(zip(*self.normalized_data.data_y))[i]

        x = range(len(self.data.data_y))

        if not self.ui.norm_plot.isChecked():
            Y_real = list(zip(*self.data.data_y))[i]

            lst_min = min(Y_real)
            lst_max = max(Y_real)

            Y_approx = [
                (y * (lst_max - lst_min) + lst_min) for y in Y_approx
                ]

        plt.plot(x, Y_real, color='red',
                 marker='o', markevery=x)
        plt.plot(x, Y_approx, color='green')

        plt.legend(['Real', 'Model'])
        
        resid_norm = max(
            list(map(
                lambda x, y: abs(x - y), Y_approx, Y_real
            ))
        )
        
        plt.gcf().canvas.set_window_title('Y' + str(i) + ', Похибка: ' + str(resid_norm))
        plt.show()

def calculate(x):
    computed =  Calculator(x[0], x[1], x[2], x[3], x[4], x[5], x[6])
    return computed.pol_pow_x1, computed.pol_pow_x2, computed.pol_pow_x3, computed.resids
def normalize(lst):
    lst_max = max(lst)
    lst_min = min(lst)
    return list(map(lambda x: ((x - lst_min) / (lst_max - lst_min)), lst))


def normalize_data_container(data_container):
    x1_t = list(zip(*data_container.data_x1))
    x2_t = list(zip(*data_container.data_x2))
    x3_t = list(zip(*data_container.data_x3))
    y_t = list(zip(*data_container.data_y))

    x1_normilized = [normalize(x) for x in x1_t]
    x2_normilized = [normalize(x) for x in x2_t]
    x3_normilized = [normalize(x) for x in x3_t]
    y_normilized = [normalize(x) for x in y_t]

    data_container.data_x1 = list(zip(*x1_normilized))
    data_container.data_x2 = list(zip(*x2_normilized))
    data_container.data_x3 = list(zip(*x3_normilized))
    data_container.data_y = list(zip(*y_normilized))




if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = App()
    win.ui.show()
    sys.exit(app.exec_())
