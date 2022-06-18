# step03 함수의 연결
# Function 의 __call__ 메서드의 입출력이 Variable 인스턴스로 통일되어 있어서 여러 함수를 연속하여 적용할 수 있음.

import numpy as np

from step01 import Variable
from step02 import Function, Square

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

def test():
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    print(y.data)