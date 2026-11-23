<<<<<<< HEAD
Individual Project Proposal

Smart Meal and Grocery Recommendation System

Student Name: Stacey Mufaro Nyirenda
Student ID: 73505

1. Problem Statement

Planning meals and deciding what groceries to buy can be difficult, especially when trying to save time, manage a budget and maintain a balanced diet.

This project aims to develop a machine learning-based system that recommends meals and grocery items based on users' food preferences, available ingredients and budget. It will also estimate calories and ingredient quantities to help users plan their meals.

2. Dataset

The project will explore datasets containing information such as:

Food and recipe names.

Ingredients and nutritional information.

Grocery items and prices.

Food categories.

Serving sizes and user preferences.

A suitable public dataset will be explored for training and testing the system.

3. Proposed Approach

A machine learning model will be developed using Python and PyTorch.

The system will analyse food preferences, available ingredients and budget information to recommend suitable meals and grocery items.

It will also provide:

Approximate calorie estimates for each meal.

Ingredient quantities in grams based on the number of servings.

Estimated nutritional information.

Grocery lists showing the ingredients that need to be purchased.

Meal suggestions that consider the user's budget and preferences.

For example, a user could enter the ingredients they have at home and the number of people they are cooking for. The system would suggest meals, calculate approximate ingredient quantities and calories, and list any additional groceries needed.

4. Success Criteria

The proposed project targets are:

Metric                      Target

Recommendation accuracy     At least 85%

Inference latency           50 ms or less

Model size                  5 MB or less

Calorie estimates          Reasonable estimates based on recipe ingredients and quantities
 
5. Model Optimisation

The project will explore optimisation techniques such as:

Pruning: Removing less important model weights or filters.

Quantization: Reducing model precision to make it smaller and potentially faster.

Knowledge distillation: Training a smaller model using knowledge from a larger model.

Different model versions will be compared based on recommendation performance, model size and inference latency.

6. Expected Outcome
=======
### **Individual Project Proposal: Systems-Optimized Handwritten Digit Recognition and Model Compression**

### 

### **Student Name: Stacey Mufaro Nyirenda**

### **Student ID: 73505**

### 

### **1. Problem Statement**

### 

### Handwritten digit recognition is a computer vision task used in applications such as postal code reading and automated form processing. This project aims to develop a Convolutional Neural Network (CNN) using PyTorch to recognise handwritten digits from 0–9. The main focus is to build an efficient and reproducible machine learning pipeline and investigate model compression techniques to reduce storage and inference costs while maintaining high accuracy.

### 

### **2. Dataset**

### 

### Dataset: MNIST (60,000 training images and 10,000 testing images).

### 

### Split: 55,000 training, 5,000 validation and 10,000 testing images, using a fixed random seed.

### 

### Preprocessing: Image normalisation, tensor conversion and configurable DataLoaders.

### 

### Experiments: Compare num\_workers \[0, 1, 2, 4, 8], batch sizes \[32, 64, 128, 256, 512] and gradient accumulation.

### 

### **3. Success Metrics**

### 

### Primary Metric: Test classification accuracy ≥ 97%.

### 

### Secondary Metrics: Training throughput (samples/s), peak memory, parameter count, model size and inference latency.

### 

### Profiling: Use torch.profiler to measure data loading, forward pass, backward pass and optimizer time.

### 

### **4. Deployment Constraints**

### 

### The final model should support efficient local CPU inference without specialised hardware.

### 

### Latency: ≤ 50 ms per image prediction (batch size 1).

### 

### Model Size: ≤ 5 MB for the final serialized model.

### 

### Accuracy: Aim to maintain ≥ 97% classification accuracy after optimisation.

### 

### **5. Planned Training and Optimization**

### 

### The pipeline will use a single YAML configuration, fixed random seeds, checkpoint saving and recovery, and MLflow tracking of parameters, metrics and artifacts.

### 

### Training experiments will compare:

### 

### Optimizers and Schedulers: Adam vs. SGD with momentum, using StepLR, CosineAnnealingLR and ReduceLROnPlateau.

### 

### Data Loading: Worker and batch-size configurations to identify performance bottlenecks.

### 

### Gradient Accumulation: Compare batch size 256 against batch size 64 with accumulation steps of 4.

### 

### Checkpoint Recovery: Interrupt training after epoch 3 and resume from the saved checkpoint.

### 

### Model compression will investigate:

### 

### Unstructured Pruning: Global magnitude-based pruning at sparsity levels \[25%, 50%, 75%, 90%].

### 

### Structured Pruning: Remove less important CNN filters using L2-norm-based pruning and fine-tuning.

### 

### Knowledge Distillation: Transfer knowledge from a larger teacher CNN to a smaller student CNN using temperature-scaled KL divergence (T = 4, α = 0.5), comparing the student against one trained from scratch.

### 

### INT8 Quantization: Apply quantization to reduce model size and evaluate CPU inference performance.

### 

### The baseline, pruned, distilled and quantized models will be compared based on accuracy, model size and inference latency.

### 

### **6. Expected Outcome**

### 

### A reproducible PyTorch training pipeline and a compact handwritten digit recognition model that maintains high accuracy while meeting the specified deployment constraints.
>>>>>>> c61fea9 (update Lab 6 individual project proposal)

The expected outcome is a simple intelligent system that helps users plan meals, estimate calories, calculate ingredient quantities and prepare grocery lists based on their preferences, available ingredients and budget. The final model should be efficient enough to run on ordinary computers while maintaining good recommendation performance.
