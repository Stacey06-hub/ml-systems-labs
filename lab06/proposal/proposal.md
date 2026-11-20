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

The expected outcome is a simple intelligent system that helps users plan meals, estimate calories, calculate ingredient quantities and prepare grocery lists based on their preferences, available ingredients and budget. The final model should be efficient enough to run on ordinary computers while maintaining good recommendation performance.
