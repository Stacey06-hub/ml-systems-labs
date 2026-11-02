
import torch
import torch.nn as nn
import time
import statistics
import os

torch.manual_seed(42)

# Small neural network
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

x = torch.randn(300, 2)

# Trace the model
traced_model = torch.jit.trace(model, x)

# Inspect execution graph
print("=== COMPUTATION GRAPH ===")
print(traced_model.graph)

# Warm-up
for _ in range(10):
    model(x)
    traced_model(x)

# Measure eager and traced execution
def measure(function):
    times = []
    for _ in range(100):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return statistics.median(times)

eager_time = measure(lambda: model(x))
traced_time = measure(lambda: traced_model(x))

print("\n=== EXECUTION TIMINGS ===")
print("Eager median:", eager_time)
print("Traced median:", traced_time)

# Tensor properties
a = torch.randn(1000, 1000)
noncontiguous = a.t()
contiguous = noncontiguous.contiguous()

print("\n=== TENSOR PROPERTIES ===")
print("Shape:", noncontiguous.shape)
print("Dtype:", noncontiguous.dtype)
print("Original contiguous:", a.is_contiguous())
print("Transposed contiguous:", noncontiguous.is_contiguous())
print("Copied contiguous:", contiguous.is_contiguous())

# Compare operation speed
def tensor_operation(t):
    return t * 2 + 1

noncontiguous_time = measure(lambda: tensor_operation(noncontiguous))
contiguous_time = measure(lambda: tensor_operation(contiguous))

print("\n=== TENSOR PERFORMANCE ===")
print("Non-contiguous median:", noncontiguous_time)
print("Contiguous median:", contiguous_time)

# Save results
os.makedirs("lab05/results", exist_ok=True)

with open("lab05/results/execution_results.txt", "w") as f:
    f.write(f"Eager median: {eager_time}\n")
    f.write(f"Traced median: {traced_time}\n")
    f.write(f"Non-contiguous median: {noncontiguous_time}\n")
    f.write(f"Contiguous median: {contiguous_time}\n")
    f.write(f"Tensor shape: {noncontiguous.shape}\n")
    f.write(f"Tensor dtype: {noncontiguous.dtype}\n")
    f.write(f"Original contiguous: {a.is_contiguous()}\n")
    f.write(f"Transposed contiguous: {noncontiguous.is_contiguous()}\n")

print("\nResults saved successfully!")
