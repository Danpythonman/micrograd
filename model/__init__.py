from typing import Callable, cast, Optional, List, Set, Tuple
import math
import random

import numpy as np


class Value:

    data: float
    label: str
    _backward: Callable[[], None]
    _prev: Set['Value']
    _op: str
    grad: float

    def __init__(
        self,
        data: float,
        label: str = '',
        _children: Tuple['Value', ...] = (),
        _op: str = ''
    ):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f'Value(data={self.data})'

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, _children=(self, other), _op='+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, _children=(self, other), _op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        # Only support int or float powers
        assert isinstance(other, (int, float))
        out = Value(self.data**other, _children=(self,), _op=f'^{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad

        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * (other**-1)

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)

        out = Value(t, _children=(self,), _op='tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        x = self.data

        out = Value(math.exp(x), _children=(self,), _op='exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()


class Neuron:

    w: List[Value]
    b: Value

    def __init__(self, nin: int):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x: List[Value]) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self) -> List[Value]:
        return self.w + [self.b]


class Layer:

    neurons: List[Neuron]

    def __init__(self, nin: int, nout: int):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x: List[Value]) -> List[Value]:
        outs = [n(x) for n in self.neurons]
        return outs

    def parameters(self) -> List[Value]:
        return [param for neuron in self.neurons for param in neuron.parameters()]


class MultiLayerPerceptron:

    layers: List[Layer]

    def __init__(self, nin: int, nouts: List[int]):
        '''
        Initialize a multi-layer perceptron with a number of inputs and a list
        representing the number of outputs of each layer.
        '''

        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x: List[Value]) -> Value | List[Value]:
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self) -> List[Value]:
        return [param for layer in self.layers for param in layer.parameters()]


def loss_mean_square_error(
    model: MultiLayerPerceptron,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: Optional[int] = None
):
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]

    inputs = np.array([[Value(x) for x in xrow] for xrow in Xb])

    ypred = np.array([model(inp) for inp in inputs])

    loss = sum((yout - ygt)**2 for ygt, yout in zip(yb, ypred))

    accuracy_list = [(ygt > 0) == (yout.data > 0) for ygt, yout in zip(yb, ypred)]

    acc = sum(accuracy_list) / len(accuracy_list)

    return cast(Value, loss), acc
