import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None
    
    # x -> func -> y 에서 y 가 볼 때 function 에 의해 creation.
    # 즉, func 가 creator
    def set_creator(self, func):
        self.creator = func
    
    def backward(self):
        f = self.creator
        if f is not None:
            x = f.input # 함수의 입력
            x.grad = f.backward(self.grad) # 함수의 backward 메서드를 호출
            x.backward() # 하나 앞 변수의 backward 메서드를 호출 (recursive)

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self) # 출력 변수에 creator 를 설정
        self.input = input
        self.output = output # 출력도 저장
        return output
    
    def forward(self, x):
        return NotImplementedError()
    
    def backward(self, gy):
        return NotImplementedError()

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy # dy/dx = 2x, chain rule
        return gx

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy # dy/dx = exp(x), chain rule
        return gx

def test01():
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    assert y.creator == C
    assert y.creator.input == b
    assert y.creator.input.creator == B
    assert y.creator.input.creator.input == a
    assert y.creator.input.creator.input.creator == A
    assert y.creator.input.creator.input.creator.input == x

def test():
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    y.grad = np.array(1.0)
    y.backward() # recursive 하게 진행
    print(x.grad)