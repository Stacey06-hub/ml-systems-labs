Lab 7: Model Compression – Pruning and Knowledge Distillation



1\. Introduction



The aim of this lab was to test model compression techniques and understand how they affect model accuracy, size, and inference speed.



The experiments covered baseline CNN training, magnitude pruning, structured pruning, dense and sparse benchmarking, knowledge distillation, and efficiency analysis.



2\. Dataset and Setup



The Fashion-MNIST dataset was divided into:



&#x20;Training: 55,000 images

&#x20;Validation: 5,000 images

&#x20;Testing: 10,000 images



A batch size of 128 was used. All experiments were performed on a CPU.



3\. Baseline CNN



An oversized CNN called `OversizedCNN` was trained for five epochs using Adam with a learning rate of 0.001.



| Metric | Result |

|---|---:|

| Parameters | 979,498 |

| Test accuracy | 92.44% |

| Model size | 3.74 MB |

| Inference latency | 1.823 ms |



The baseline model was used as the reference for the compression experiments.



4\. Magnitude-Based Pruning



Global and per-layer pruning were tested at 0%, 25%, 50%, 75%, 90%, and 95% sparsity. Each model was fine-tuned for three epochs with pruning masks fixed.



| Sparsity | Global validation accuracy | Per-layer validation accuracy |

|---|---:|---:|

| 0% | 91.80% | 91.88% |

| 25% | 91.52% | 90.74% |

| 50% | 91.40% | 90.86% |

| 75% | 91.12% | 89.38% |

| 90% | 89.62% | 86.68% |

| 95% | 8.90% | 84.78% |



Per-layer pruning performed much better at 95% sparsity. Global pruning dropped to 8.90% validation accuracy, while per-layer pruning achieved 84.78%.



5\. Structured Pruning



Structured pruning was tested by removing filters from convolutional layers and rebuilding smaller CNN models.



| Target filter removal | Parameters | Model size | Test accuracy | Latency |

|---|---:|---:|---:|---:|

| Baseline | 979,498 | 3.74 MB | 92.44% | 0.866 ms |

| 10% | 871,674 | 3.33 MB | 92.33% | 0.827 ms |

| 25% | 721,330 | 2.76 MB | 91.85% | 0.507 ms |

| 50% | 472,378 | 1.81 MB | 87.58% | 0.419 ms |



The filters were physically removed, creating smaller models. The 25% target removal model reduced the parameter count by 26.36% while maintaining 91.85% test accuracy. At 50% target removal, the model was smaller and faster, but accuracy dropped to 87.58%.



6\. Dense and Sparse Benchmarking



Models were tested on CPU using batch size 1, 20 warm-up runs, and 200 timed runs.



| Model | Sparsity | Size | Latency |

|---|---:|---:|---:|

| Baseline | 0% | 3.74 MB | 0.886 ms |

| Global | 25% | 3.74 MB | 0.730 ms |

| Global | 50% | 3.74 MB | 0.711 ms |

| Global | 75% | 3.74 MB | 0.803 ms |

| Global | 90% | 3.74 MB | 0.862 ms |

| Global | 95% | 3.74 MB | 0.749 ms |

| Per-layer | 25% | 3.74 MB | 0.675 ms |

| Per-layer | 50% | 3.74 MB | 0.757 ms |

| Per-layer | 75% | 3.74 MB | 0.613 ms |

| Per-layer | 90% | 3.74 MB | 0.630 ms |

| Per-layer | 95% | 3.74 MB | 0.662 ms |



Unstructured sparsity did not reduce model size or consistently improve inference speed. This happened because zero weights remained stored in dense tensors and were still processed by the CPU.



The baseline training script measured 1.823 ms, while the separate benchmark measured 0.886 ms. The separate benchmark used the same timing method for all models in that comparison.



7\. Knowledge Distillation



A large CNN was used as the teacher and a smaller CNN as the student. The distillation temperature was 4, alpha was 0.5, and both student experiments were trained for five epochs.



| Model | Parameters | Size | Test accuracy |

|---|---:|---:|---:|

| Teacher | 979,498 | 3.74 MB | 92.44% |

| Standalone student | 80,602 | 0.31 MB | 89.65% |

| Distilled student | 80,602 | 0.31 MB | 89.44% |



The student model was much smaller than the teacher. However, knowledge distillation did not improve accuracy in this experiment. The standalone student performed slightly better.



8\. Efficiency Analysis



The experiments considered four efficiency pillars:



\- Data: The same Fashion-MNIST training, validation, and test splits were used for consistent comparisons.

\- Model: The student model reduced the parameter count from 979,498 to 80,602 and model size from 3.74 MB to 0.31 MB.

\- Compute: Physical structured pruning reduced inference latency. Unstructured sparsity did not consistently improve speed on dense CPU operations.

\- Energy: Direct energy consumption in watts or joules was not measured because a reliable power measurement was unavailable.



The results show that smaller models can reduce storage requirements and physical filter removal can improve inference efficiency. Actual energy savings would need to be measured separately.



9\. Conclusion



This lab showed that pruning and knowledge distillation can reduce model size while maintaining useful classification accuracy.



Per-layer pruning performed better than global pruning at high sparsity. Physical structured pruning reduced model size and inference latency, with the 25% target removal model achieving 91.85% test accuracy.



The compact student model reduced model size from 3.74 MB to 0.31 MB and achieved 89.65% test accuracy when trained alone. Knowledge distillation did not improve its accuracy in this experiment.



Overall, physical structured pruning and the compact student model showed useful ways to reduce model size and computation. Direct energy consumption was not measured and remains a limitation of this experiment.

