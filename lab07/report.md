Lab 7: Model Compression – Pruning and Knowledge Distillation



1\. Introduction



The aim of this lab was to test model compression techniques and see how they affect model accuracy, size, and speed.



The experiments used Fashion-MNIST and covered baseline training, pruning, benchmarking, knowledge distillation, and efficiency analysis.



2\. Dataset and Setup



The Fashion-MNIST dataset was divided into:



\- Training: 55,000 images

\- Validation: 5,000 images

\- Testing: 10,000 images



A batch size of 128 was used. All experiments were performed on a CPU.



3\. Baseline CNN



An oversized CNN called OversizedCNN was trained for five epochs using Adam with a learning rate of 0.001.



Results:



| Metric | Result |

|---|---:|

| Parameters | 979,498 |

| Test accuracy | 92.44% |

| Model size | 3.74 MB |

| Inference latency | 1.823 ms |



The baseline model achieved 92.44% test accuracy and was used for comparison with the compressed models.



4\. Magnitude-Based Pruning



Global and per-layer pruning were tested at 0%, 25%, 50%, 75%, 90%, and 95% sparsity. Each model was fine-tuned for three epochs.



Global pruning results:



| Sparsity | Validation accuracy |

|---|---:|

| 0% | 91.80% |

| 25% | 91.52% |

| 50% | 91.40% |

| 75% | 91.12% |

| 90% | 89.62% |

| 95% | 8.90% |



Per-layer pruning results:



| Sparsity | Validation accuracy |

|---|---:|

| 0% | 91.88% |

| 25% | 90.74% |

| 50% | 90.86% |

| 75% | 89.38% |

| 90% | 86.68% |

| 95% | 84.78% |



Per-layer pruning performed better at high sparsity. At 95%, it achieved 84.78% accuracy, while global pruning dropped to 8.90%.



5\. Structured Pruning



Structured pruning was tested at 10%, 25%, and 50%.



| Target pruning | Zero filters |

|---|---:|

| 10% | 19 of 192 |

| 25% | 48 of 192 |

| 50% | 96 of 192 |



The filters were masked but not physically removed. Therefore, the parameter count remained 979,498.



6\. Dense and Sparse Benchmarking



The models were tested on CPU using batch size 1, 20 warm-up runs, and 200 timed runs.



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



The results show that sparsity did not consistently improve inference speed or reduce model size. This is because zero weights were still stored and processed in dense tensors.



The baseline training script measured 1.823 ms, while the separate benchmark measured 0.886 ms. The benchmark results above use the same measurement method for all compared models.



7\. Knowledge Distillation



A large CNN was used as the teacher, and a smaller CNN was trained as the student.



The distillation settings were:



\- Temperature: 4

\- Alpha: 0.5

\- Training epochs: 5



Results:



| Model | Parameters | Size | Test accuracy |

|---|---:|---:|---:|

| Teacher | 979,498 | 3.74 MB | 92.44% |

| Standalone student | 80,602 | 0.31 MB | 89.65% |

| Distilled student | 80,602 | 0.31 MB | 89.44% |



The student model was much smaller than the teacher. However, distillation did not improve accuracy. The standalone student performed slightly better.



8\. Efficiency Analysis



Data efficiency:

The same Fashion-MNIST dataset and data loader were used for all models. This helped keep the experiments consistent.



Model efficiency:

The student model reduced the parameter count from 979,498 to 80,602 and the model size from 3.74 MB to 0.31 MB.



Compute efficiency:

All experiments were performed on CPU. Unstructured pruning did not consistently improve inference speed because the zero weights were still stored in dense tensors.



Energy efficiency:

Direct energy consumption was not measured. The smaller student model may use fewer resources, but actual energy savings would need to be measured.



Summary:



| Pillar | Finding |

|---|---|

| Data | Same dataset and data loader used |

| Model | Student model was much smaller |

| Compute | Sparse models did not show consistent speed improvements |

| Energy | Direct measurements were not available |



9\. Conclusion



This lab showed that pruning and knowledge distillation can help reduce model size.



The baseline achieved 92.44% test accuracy. Per-layer pruning performed better than global pruning at high sparsity. However, unstructured pruning did not reduce model size or consistently improve speed.



The student model reduced the size from 3.74 MB to 0.31 MB while achieving 89.65% accuracy when trained alone.



Overall, the smaller student model gave the clearest reduction in model size. Further work could include physically removing filters and measuring actual energy consumption.

