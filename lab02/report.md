# Lab 2: Neural Network from Scratch

## 1. Objective

The goal was to build a neural network using NumPy, test its gradients, train it on digit images, and measure its speed.

## 2. Method

I used the digits dataset. I scaled the inputs by dividing by 16 and split the data into 70% training and 30% testing data.

The network has two hidden layers using ReLU and an output layer using softmax. I tested hidden layer sizes of 16 and 64, with learning rates of 0.001, 0.01, and 0.1. Each model trained for 50 epochs using batches of 32.

## 3. Gradient Check

The initial loss was about 2.3026, as expected. The gradient check passed for 9 out of 10 weights, meeting the required result.

## 4. Training Results

| Hidden width | Learning rate | Final loss | Test accuracy |
|---:|---:|---:|---:|
| 16 | 0.001 | 2.3025 | 10.19% |
| 16 | 0.01 | 2.3025 | 10.19% |
| 16 | 0.1 | 1.5970 | 32.59% |
| 64 | 0.001 | 2.3025 | 10.19% |
| 64 | 0.01 | 2.3023 | 10.19% |
| 64 | 0.1 | 0.1684 | 93.89% |

The loss graph is saved as `results/loss_curves.png`. The full results are in `results/training_results.csv`.

## 5. Performance

The backward pass was slower than the forward pass in all tests because it calculates gradients for the network's layers. Wider networks also generally took longer to run.

## 6. Conclusion

The best model used a hidden width of 64 and a learning rate of 0.1. It achieved **93.89% test accuracy**.

The lower learning rates did not learn much in 50 epochs. The gradient check passed the requirement, and the timing tests showed that the backward pass took longer than the forward pass.git status