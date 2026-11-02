
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, power):
        if not isinstance(power, (int, float)):
            raise TypeError("Power must be a number")

        out = Value(self.data ** power, (self,), f"**{power}")

        def backward():
            self.grad += (
                power * self.data ** (power - 1) * out.grad
            )

        out._backward = backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "ReLU")

        def backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = backward
        return out

    def sum(self):
        # Values are scalars, so their sum is themselves.
        return self

    def backward(self):
        topo = []
        visited = set()

        def build_topology(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topology(child)
                topo.append(node)

        build_topology(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
