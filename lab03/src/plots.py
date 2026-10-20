import os
import matplotlib.pyplot as plt

# Results from Lab 3
models = ["MLP", "CNN", "Attention"]
parameters = [235146, 20490, 70794]
accuracy = [86.47, 88.86, 83.71]
latency_ms = [0.087, 0.266, 0.373]  # Batch size 1

os.makedirs("lab03/results", exist_ok=True)

# Plot 1: Accuracy vs. number of parameters
plt.figure(figsize=(8, 5))
plt.scatter(parameters, accuracy)

for i, name in enumerate(models):
    plt.annotate(name, (parameters[i], accuracy[i]), xytext=(5, 5),
                 textcoords="offset points")

plt.xlabel("Number of Parameters")
plt.ylabel("Test Accuracy (%)")
plt.title("Fashion-MNIST: Accuracy vs. Parameters")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lab03/results/accuracy_vs_parameters.png", dpi=150)
plt.close()

# Plot 2: Accuracy vs. batch-1 inference latency
plt.figure(figsize=(8, 5))
plt.scatter(latency_ms, accuracy)

for i, name in enumerate(models):
    plt.annotate(name, (latency_ms[i], accuracy[i]), xytext=(5, 5),
                 textcoords="offset points")

plt.xlabel("Batch-1 Inference Latency (ms)")
plt.ylabel("Test Accuracy (%)")
plt.title("Fashion-MNIST: Accuracy vs. Inference Latency")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lab03/results/accuracy_vs_latency.png", dpi=150)
plt.close()

print("Both plots saved in lab03/results/")