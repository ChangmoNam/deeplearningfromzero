import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data

def test():
    data = np.array(1.0)
    x = Variable(data)
    print(x.data, x.data.ndim)
