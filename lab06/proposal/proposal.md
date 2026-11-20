### **Individual Project Proposal**

##### 

##### **Waste Classification Using Deep Learning**



**Student Name: Stacey Mufaro Nyirenda**  

**Student ID: 73505**



###### 1\. **Problem Statement**



Waste management is an important environmental challenge. Incorrect disposal of waste makes recycling more difficult and contributes to environmental pollution.



This project proposes developing a deep learning model that can automatically classify waste images into different categories. The aim is to make waste sorting easier and support more efficient recycling.



###### 2\. Dataset



The project will use the TrashNet dataset, which contains images belonging to six waste categories:



\- Cardboard

\- Glass

\- Metal

\- Paper

\- Plastic

\- Trash



The images will be prepared and divided into training, validation and testing datasets.



###### **3. Proposed Model and Method**



A Convolutional Neural Network (CNN) will be developed using PyTorch.



The model will learn visual features from waste images and predict the correct waste category.



The project will involve:

\- Loading and preprocessing the dataset.

\- Building and training a CNN.

\- Evaluating model performance.

\- Applying model optimization techniques.

\- Comparing the model before and after optimization.



###### **4. Success Criteria**



The following are the proposed project targets:



| Metric | Target |

|---|---|

| Test accuracy | At least 85% |

| Inference latency | 50 ms or less per image |

| Model size | 5 MB or less |



These are target values, not achieved results.



###### **5. Model Optimization**



The project plans to investigate:



Pruning: Removing less important model weights to reduce model complexity.

Quantization: Reducing numerical precision to make the model smaller and potentially faster.



The optimized model will be evaluated against the original model to determine the effect on accuracy, model size and inference latency.



###### **6. Initial Code Preview**



The following is an initial example of the proposed CNN architecture:



import torch.nn as nn



model = nn.Sequential(

&#x20;   nn.Conv2d(3, 16, kernel\_size=3, padding=1),

&#x20;   nn.ReLU(),

&#x20;   nn.MaxPool2d(2),



&#x20;   nn.Conv2d(16, 32, kernel\_size=3, padding=1),

&#x20;   nn.ReLU(),

&#x20;   nn.MaxPool2d(2),



&#x20;   nn.AdaptiveAvgPool2d((1, 1)),

&#x20;   nn.Flatten(),

&#x20;   nn.Linear(32, 6)

)



This is a preliminary architecture preview. The final model may be adjusted during implementation.



###### **7. Expected Outcome**



The expected outcome is a trained waste classification model that can identify six categories of waste from images.



The project will also investigate whether pruning and quantization can reduce model size and inference latency while maintaining acceptable classification accuracy.



