"""Scalar reverse-mode automatic differentiation for Lab 05."""
import math
from numbers import Real


class Value:
    """A scalar, its gradient, parent nodes and a local backward rule."""

    def __init__(self, data, parents=(), op='', label=''):
        self.data = float(data)
        self.grad = 0.0
        # A tuple retains repeated operands, e.g. both inputs of x * x.
        self.parents = tuple(parents)
        self.op = op
        self.label = label
        self._backward = lambda: None

    def __repr__(self):
        return f'Value(data={self.data}, grad={self.grad})'

    @staticmethod
    def _value(other):
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other):
        other = self._value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def rule():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = rule
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = self._value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def rule():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = rule
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, power):
        if not isinstance(power, Real):
            raise TypeError('Power must be a constant real number.')
        result = self.data ** power
        if isinstance(result, complex):
            raise ValueError('Value supports real outputs only.')
        out = Value(result, (self,), f'**{power}')

        def rule():
            derivative = 0.0 if power == 0 else power * self.data ** (power - 1)
            self.grad += derivative * out.grad

        out._backward = rule
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -self._value(other)

    def __rsub__(self, other):
        return self._value(other) + -self

    def __truediv__(self, other):
        return self * self._value(other) ** -1

    def relu(self):
        out = Value(max(0.0, self.data), (self,), 'ReLU')

        def rule():
            self.grad += float(self.data > 0) * out.grad

        out._backward = rule
        return out

    def sigmoid(self):
        if self.data >= 0:
            value = 1 / (1 + math.exp(-self.data))
        else:
            exp_x = math.exp(self.data)
            value = exp_x / (1 + exp_x)
        out = Value(value, (self,), 'sigmoid')

        def rule():
            self.grad += value * (1 - value) * out.grad

        out._backward = rule
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')

        def rule():
            self.grad += out.grad / self.data

        out._backward = rule
        return out

    def sum(self):
        # A scalar is a one-element sum; preserve a node for the requested expression.
        return self + 0.0

    def topological_order(self):
        order, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
            elif node not in visited:
                visited.add(node)
                stack.append((node, True))
                stack.extend((parent, False) for parent in reversed(node.parents))
        return order

    def backward(self):
        """Propagate on a fresh graph; clear leaf grads before a new training step."""
        order = self.topological_order()
        self.grad = 1.0
        for node in reversed(order):
            node._backward()
