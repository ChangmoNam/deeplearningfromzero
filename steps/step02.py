import numpy as np

from step01 import Variable

class Function01:
    def __call__(self, input):
        x = input.data
        y = x**2
        output = Variable(y)
        return output

def test01():
    x = Variable(np.array(10))
    f = Function01()
    y = f(x)

    print(type(y))
    print(y.data)

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output
    
    def forward(self, x):
        raise NotImplementedError() # 예외 발생 -> 상속해서 구현해야 함을 알려줌.

class Square(Function):
    def forward(self, x):
        return x**2 # __call__ 메서드가 계승되므로 logic 만 작성하면 okay.


def test():
    x = Variable(np.array(10))
    f = Square()
    y = f(x)
    print(type(y))
    print(y.data)