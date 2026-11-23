

Lab 7: Efficiency Analysis



1\. Introduction



This analysis evaluates the efficiency of the baseline CNN, pruned models, and knowledge-distilled student model. It considers four pillars: data, model, compute, and energy.



2\. Data Efficiency



The experiments used the Fashion-MNIST dataset.



| Dataset split | Number of images |

|---|---:|

| Training | 55,000 |

| Validation | 5,000 |

| Testing | 10,000 |

| Total | 70,000 |



The training data was loaded using the existing Lab 3 data loader. A batch size of 128 was used.



Reusing the same dataset and data-loading process helped keep the experiments consistent. All models were evaluated on the same test dataset, making their accuracy results comparable.



3\. Model Efficiency



Three model approaches were evaluated:



| Model | Parameters | Model size | Test accuracy |

|---|---:|---:|---:|

| Baseline CNN | 979,498 | 3.74 MB | 92.44% |

| Standalone student | 80,602 | 0.31 MB | 89.65% |

| Distilled student | 80,602 | 0.31 MB | 89.44% |



The student model has significantly fewer parameters than the baseline CNN. Its model size is also smaller, while maintaining reasonably close test accuracy.



The standalone student achieved 89.65% accuracy, while the distilled student achieved 89.44%. In this experiment, knowledge distillation did not improve the student's accuracy. The difference was 0.21 percentage points.



Pruning results



Magnitude pruning was tested at sparsity levels of 25%, 50%, 75%, 90%, and 95%.



Global pruning maintained 91.12% validation accuracy at 75% sparsity but dropped to 8.90% at 95%.



Per-layer pruning performed better at high sparsity, achieving 84.78% validation accuracy at 95% sparsity.



However, unstructured pruning did not reduce the number of model parameters or the saved model size because zero weights remained stored in dense tensors.



4\. Compute Efficiency



All experiments were performed using a CPU.



The dense and sparse benchmark used batch size 1, 20 warm-up iterations, and 200 timed iterations.



| Model | Sparsity | Model size | Inference latency |

|---|---:|---:|---:|

| Baseline CNN | 0% | 3.74 MB | 0.886 ms |

| Global pruning | 25% | 3.74 MB | 0.730 ms |

| Global pruning | 50% | 3.74 MB | 0.711 ms |

| Global pruning | 75% | 3.74 MB | 0.803 ms |

| Global pruning | 90% | 3.74 MB | 0.862 ms |

| Global pruning | 95% | 3.74 MB | 0.749 ms |

| Per-layer pruning | 25% | 3.74 MB | 0.675 ms |

| Per-layer pruning | 50% | 3.74 MB | 0.757 ms |

| Per-layer pruning | 75% | 3.74 MB | 0.613 ms |

| Per-layer pruning | 90% | 3.74 MB | 0.630 ms |

| Per-layer pruning | 95% | 3.74 MB | 0.662 ms |



The results show that sparse models did not consistently achieve lower inference latency than the baseline.



This is because unstructured pruning creates zero weights but does not automatically change the dimensions of the weight tensors. Dense CPU operations may still process those zero values.



Special sparse storage formats and sparse computation kernels are needed to benefit more directly from unstructured sparsity.



The compact student model also offers a compute advantage through its much smaller number of parameters, although a separate student latency benchmark would be needed to quantify that advantage.



5\. Energy Efficiency



The experiments were performed on a CPU-only system. Direct energy consumption in joules or watts was not measured because a reliable power measurement was not available.



However, reducing model parameters and model size can reduce the amount of computation and memory required during inference.



The student model contains 80,602 parameters compared with 979,498 in the baseline CNN. This makes it a promising option for lower-resource environments.



The sparse models did not reduce their saved file size, so their energy savings cannot be assumed from sparsity alone.



Future experiments could use a power-monitoring tool or an external power meter to measure actual energy consumption during training and inference.



6\. Summary of the Four Efficiency Pillars



| Pillar | Findings | Efficiency implication |

|---|---|---|

| Data | 55,000 training, 5,000 validation, and 10,000 test images | Consistent data loading supported fair comparisons |

| Model | Student model has 80,602 parameters and a size of 0.31 MB | Smaller model requires less storage |

| Compute | CPU inference benchmark showed latency differences between models | Unstructured sparsity did not consistently improve latency |

| Energy | Direct power measurements were not available | Smaller models may reduce resource requirements, but energy savings need measurement |



7\. Conclusion



The experiments show that model compression can reduce model size and parameter count while maintaining useful classification accuracy.



The compact student model reduced the parameter count from 979,498 to 80,602 and model size from 3.74 MB to 0.31 MB. Its standalone test accuracy was 89.65%, compared with 92.44% for the baseline CNN.



Magnitude pruning demonstrated that per-layer pruning retained better validation accuracy than global pruning at very high sparsity. However, unstructured pruning did not reduce model size or consistently improve CPU inference latency.



Knowledge distillation did not improve accuracy in this experiment, although the student model remained much smaller than the teacher.



Overall, the compact student model provides the clearest reduction in model size and parameter count. Further work could investigate physically removing filters, sparse inference kernels, and direct energy measurement.





