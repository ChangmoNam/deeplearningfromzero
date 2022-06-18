# numerical differentiation 수치미분
# 구현이 쉽지만 계산량이 많음. 수백만 개 parameter 사용해야 할 때 계산이 어려움.
# back propagation 사용하게 됨.

import numpy as np

from step01 import Variable
from step02 import Square
from step03 import Exp

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps) # {f(x+h) - f(x-h)} / 2h <- centered difference

def test01():
    f = Square()
    x = Variable(np.array(2.0))
    dy = numerical_diff(f, x)
    print(dy)

def test_func(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x))) # exp(x^2)^2
    
def test02():
    x = Variable(np.array(0.5))
    dy = numerical_diff(test_func, x)
    print(dy) # x 를 0.5 변화시키면 y는 3.29744... 가 변한다는 의미