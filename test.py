import re
def read_file():
    with open('t_input.txt','r') as f:
        for line in f:
            line = line.split(' ')
            print(line)
            print(float(line[-1]))
read_file()