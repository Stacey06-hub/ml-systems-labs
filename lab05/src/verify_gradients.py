
import torch
from autodiff import Value

# Test values
a = 2.0
b = 3.0
c = 1.0

# 1. Our automatic differentiation engine
va = Value(a)
vb = Value(b)
vc = Value(c)

y = ((va * vb + vc) ** 2).relu().sum()
y.backward()

manual_grads = [va.grad, vb.grad, vc.grad]

# 2. PyTorch autograd
ta = torch.tensor(a, dtype=torch.float64, requires_grad=True)
tb = torch.tensor(b, dtype=torch.float64, requires_grad=True)
tc = torch.tensor(c, dtype=torch.float64, requires_grad=True)

ty = torch.relu((ta * tb + tc) ** 2)
ty.backward()

torch_grads = [ta.grad.item(), tb.grad.item(), tc.grad.item()]

# 3. Numerical gradients using central differences
def function(a, b, c):
    return max(0.0, (a * b + c) ** 2)

def numerical_gradient(index, values, eps=1e-6):
    plus = values.copy()
    minus = values.copy()
    plus[index] += eps
    minus[index] -= eps
    return (function(*plus) - function(*minus)) / (2 * eps)

values = [a, b, c]
numerical_grads = [
    numerical_gradient(i, values) for i in range(3)
]

# 4. Compare results
print("Our gradients:       ", manual_grads)
print("PyTorch gradients:   ", torch_grads)
print("Numerical gradients: ", numerical_grads)

difference = max(
    abs(x - z) for x, z in zip(manual_grads, torch_grads)
)

print("Maximum absolute difference:", difference)
print("Gradient check:", "PASS" if difference < 1e-6 else "FAIL")

assert difference < 1e-6
assert max(abs(x-z) for x, z in zip(manual_grads, numerical_grads)) < 1e-6
print("All gradient checks passed!")
