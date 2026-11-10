\# Lab 5 – Neural Networks, Automatic Differentiation and PyTorch Performance



\## 1. Goal



The goal of this lab was to understand how neural networks work internally. I explored automatic differentiation, gradient calculation, PyTorch execution, and tensor performance. I also compared NumPy and PyTorch to see how they perform when training a neural network.



\## 2. Method



\### Experiment 1: Building an Automatic Differentiation Engine



I created a `Value` class in Python to store numerical values and their gradients. I implemented addition, multiplication, power, and ReLU operations. I also added backward functions to calculate gradients using the chain rule and built a computation graph to perform backpropagation.



\### Experiment 2: Checking Gradients



I tested my automatic differentiation engine by comparing it with PyTorch autograd and numerical differentiation. I used three input values: a = 2, b = 3, and c = 1. The function multiplied two values, added the third, squared the result, and applied ReLU. Finally, I compared the gradients from all three methods.



\### Experiment 3: PyTorch Execution and Tensor Performance



I created a small neural network and used `torch.jit.trace()` to generate a traced version. I compared the execution time of the normal and traced models. I also created a 1000 × 1000 tensor, transposed it, and compared operations on the non-contiguous tensor and its contiguous copy.



\### Experiment 4: NumPy vs PyTorch Training



I trained a small neural network using NumPy and PyTorch with the `make\_moons` dataset. Both networks had the same architecture, initial weights, and training settings. I trained each model for 100 epochs and repeated the experiment five times. I then compared training time, test loss, and accuracy.



\## 3. Results



\### Experiment 1: Automatic Differentiation



My `Value` class created a computation graph and calculated gradients using the backward pass. It supported the main mathematical operations and applied the chain rule to calculate gradients.



\### Experiment 2: Checking Gradients



The expected gradients were 42 for a, 28 for b, and 14 for c. The code compared these values with the results from PyTorch and numerical differentiation. The gradient checks were designed to pass when the differences were smaller than 0.000001.



\### Experiment 3: PyTorch Execution and Tensor Performance



The eager execution time was 0.00004460 seconds, while the traced execution time was 0.00004155 seconds. The non-contiguous tensor operation took 0.00039580 seconds, compared with 0.00032870 seconds for the contiguous tensor. Therefore, the traced model and contiguous tensor were slightly faster in my experiment.



\### Experiment 4: NumPy vs PyTorch Training



NumPy had a median training time of 0.01568 seconds, while PyTorch took 0.07371 seconds. Both implementations achieved almost identical test loss (0.55022) and accuracy (81.67%). NumPy was faster for this small network.



\## 4. Conclusion



1\. I learned how automatic differentiation works and how gradients are calculated using the chain rule and backpropagation.



2\. I learned that PyTorch execution mode and tensor memory layout can affect performance. In my experiments, traced execution and contiguous tensors were slightly faster.



3\. I compared NumPy and PyTorch and found that NumPy was faster for my small network, while both implementations achieved almost the same accuracy and test loss.

