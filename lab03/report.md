\# Lab 3: Comparing Neural Network Models



\## 1. Goal



The goal of this lab was to compare three models on the Fashion-MNIST dataset: an MLP, a CNN, and an attention model.



I compared their number of parameters, computational cost, prediction speed, and test accuracy.



\## 2. Setup



\- Dataset: Fashion-MNIST

\- Training images: 55,000

\- Validation images: 5,000

\- Test images: 10,000

\- Batch size: 128

\- Training epochs: 5

\- Optimizer: Adam

\- Learning rate: 0.001

\- Device: CPU



The images were converted to tensors and normalized using a mean of 0.2860 and a standard deviation of 0.3530.



\## 3. Results



| Model | Parameters | Test Accuracy | Batch-1 Time | Batch-128 Time |

|---|---:|---:|---:|---:|

| MLP | 235,146 | 86.47% | 0.087 ms | 0.781 ms |

| CNN | 20,490 | 88.86% | 0.266 ms | 27.918 ms |

| Attention | 70,794 | 83.71% | 0.373 ms | 13.242 ms |



The CNN had the best accuracy at 88.86%. It also had the fewest parameters.



The MLP was the fastest when making a prediction for one image. The attention model was slower than the MLP for one image and had the lowest accuracy.



The times are median CPU measurements. I measured 100 runs for batch size 1 and 20 runs for batch size 128 after a short warm-up.



\## 4. FLOPs and Model Cost



FLOPs measure how many calculation steps a model needs. Here, one multiplication and one addition count as about two FLOPs. These calculations do not include bias operations or activation functions.



\### MLP



The first linear layer has 784 inputs and 256 outputs.



FLOPs = 2 x 784 x 256 = 401,408



The other layers need:



\- Second layer: 2 x 256 x 128 = 65,536

\- Third layer: 2 x 128 x 10 = 2,560



The estimated total is:



\*\*469,504 FLOPs\*\*



THOP also estimated about 469,504 FLOPs for the full MLP. This matches the calculation above.



\### CNN



For the first convolution layer:



FLOPs = 2 x 3 x 3 x 1 x 16 x 28 x 28



= 225,792



For the second convolution layer:



FLOPs = 2 x 3 x 3 x 16 x 32 x 14 x 14



= 1,806,336



For the final linear layer:



FLOPs = 2 x (32 x 7 x 7) x 10



= 31,360



The estimated total is:



\*\*2,063,488 FLOPs\*\*



THOP gave about 2,063,488 FLOPs for the CNN. This matches the hand calculation.



\### Attention Model



The image is split into 16 patches. Each patch contains 49 pixel values.



The patch projection needs:



FLOPs = 2 x 16 x 49 x 64



= 100,352



For the Q, K, and V projections in one Transformer layer:



FLOPs = 2 x 16 x 64 x (3 x 64)



= 393,216



For calculating attention scores:



FLOPs = 2 x 16 x 16 x 64



= 32,768



Calculating the attention-weighted values needs about another 32,768 FLOPs.



THOP reported 50,816 MACs, or about 101,632 FLOPs, for the attention model. However, THOP does not count all the operations inside the Transformer layers.



Because of this, its total is too low. This shows that profiling tools may miss some calculations. We should compare tool results with hand calculations.



\## 5. Trade-off Graphs



\### Accuracy vs. Number of Parameters



!\[Accuracy vs. Parameters](results/accuracy\_vs\_parameters.png)



The CNN had the best accuracy and the fewest parameters. The MLP had the most parameters but did not have the best accuracy.



This shows that a larger model is not always better.



\### Accuracy vs. Inference Time



!\[Accuracy vs. Latency](results/accuracy\_vs\_latency.png)



The MLP was the fastest for one image. The CNN had the best accuracy but took longer to make a prediction. The attention model was the slowest for one image in this test.



The results may be different on other computers or with different software.



\## 6. How the Models Work



The \*\*MLP\*\* turns the image into a long list of numbers and processes it using linear layers. This makes it simple and fast for one image, but it does not directly use the image's two-dimensional structure.



The \*\*CNN\*\* looks at small parts of the image using filters. The same filters are used across the image. This helps the model find shapes and patterns while using fewer parameters.



The \*\*attention model\*\* splits the image into patches and lets the patches exchange information. This helps the model compare different parts of the image, but it also requires extra calculations.



In this experiment, the CNN gave the best balance of accuracy and model size. The MLP was best for single-image speed. The attention model did not perform as well after five training epochs.



\## 7. Model Size



The lab target was around 100,000 to 150,000 parameters for each model. However, the specified model designs produced these results:



\- MLP: 235,146 parameters

\- CNN: 20,490 parameters

\- Attention: 70,794 parameters



I kept the specified model designs and reported the parameter counts from the actual models.



\## 8. Conclusion



The CNN achieved the best test accuracy at 88.86% and had the fewest parameters. The MLP was the fastest for one image, at 0.087 ms.



Each model has different strengths. Choosing a model means balancing accuracy, size, and speed. It is also important to understand what a profiling tool counts when measuring computational cost.

