
import re
from copy import deepcopy
from numpy import array

class DataLoader:
    def __init__(self, input_file_name='', dim_x1=0, dim_x2=0, dim_x3=0, dim_y=0, selection_range=0,
                       data_x1=[], data_x2=[], data_x3=[], data_y=[]):
        self.ERROR = False
        
        if input_file_name:
            self.data_x1 = []
            self.data_x2 = []
            self.data_x3 = []
            self.data_y = []
            self.fill_from_file(input_file_name, dim_x1, dim_x2, dim_x3, dim_y, selection_range)
            
        else:
            self.data_x1 = deepcopy(data_x1)
            self.data_x2 = deepcopy(data_x2)
            self.data_x3 = deepcopy(data_x3)
            self.data_y = deepcopy(data_y)

    def fill_from_file(self, input_file_name, dim_x1, dim_x2, dim_x3, dim_y, selection_range):
        f = open(input_file_name, "r")
        buff = []
        for i in range(0, selection_range):
            line = f.readline()
            if line == "":
                print("Your input file have empty line")
                self.ERROR = True
                return 0
            line_list = line.split(' ')
            conv_line = []
            for item in line_list:
                conv_line.append(float(item))
            
            buff.append(conv_line)

        buff = list(map(list, zip(*buff)))
        # read X1
        buff1 = array(buff)
        if buff1.shape[0] < dim_x1+dim_x2+dim_x3+dim_y:
            print("Your input file don't have enough colomns")
            self.ERROR = True
            return 0
            
        for i in range(0, dim_x1):
            self.data_x1.append(buff[i])
            
        for i in range(0, dim_x2):
            self.data_x2.append(buff[dim_x1+i])
            
        for i in range(0, dim_x3):
            self.data_x3.append(buff[dim_x1+dim_x2+i])
            
        for i in range(0, dim_y):
            self.data_y.append(buff[dim_x1+dim_x2+dim_x3+i])

        self.data_x1 = list(map(list, zip(*self.data_x1)))
        self.data_x2 = list(map(list, zip(*self.data_x2)))
        self.data_x3 = list(map(list, zip(*self.data_x3)))
        self.data_y = list(map(list, zip(*self.data_y)))

