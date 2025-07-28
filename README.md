# Medical Vision-Language Model Benchmark Results

## Task Performance Comparison

### 1. Image Classification
| Metric             | MedGemma       | Qwen 2.5       | Lingshu      |
|--------------------|---------------|---------------|-------------|
| F1-Macro           | 0.010         | **0.088**     | 0.063       |
| Accuracy-Macro     | 0.143         | **0.209**     | 0.157       |
| Precision-Macro    | 0.119         | **0.171**     | 0.124       |
| Recall-Macro       | 0.143         | **0.209**     | 0.157       |

**Leader**: Qwen 2.5 (top in all metrics)

---

### 2. Visual Question Answering (VQA)
| Metric             | MedGemma       | Qwen 2.5       | Lingshu      |
|--------------------|---------------|---------------|-------------|
| BLEU               | 0.524         | 0.430         | **0.532**   |
| F1                 | 0.538         | 0.443         | **0.550**   |
| ClosedQ Accuracy   | 0.737         | 0.602         | **0.725**   |
| OpenQ Accuracy     | 0.235         | 0.215         | **0.245**   |

**Leader**: Lingshu (best in 3/4 metrics)

---

### 3. Bounding Box Detection
| Metric             | MedGemma       | Qwen 2.5       | Lingshu      |
|--------------------|---------------|---------------|-------------|
| Mean IoU           | N/A           | **0.0875**    | 0.0478      |
| Samples Processed  | -             | Full dataset  | 213/500     |

**Leader**: Qwen 2.5 (only complete implementation)

---

### 4. Hallucination Rates
| Dataset            | MedGemma       | Qwen 2.5       | Lingshu      |
|--------------------|---------------|---------------|-------------|
| General (500 samples) | 88%         | 91.6%         | **92.3%**   |
| Medical (VQA-RAD)  | **93.8%**     | 90.0%         | 93.1%       |

**Leaders**:  
- General: Lingshu  
- Medical: MedGemma  

## 🏆 Overall Recommendations
1. **For dermatology applications**: Qwen 2.5  
   - Strongest bounding box detection (critical for lesion localization)  
   - Competitive image classification  

2. **For diagnostic Q&A systems**: Lingshu  
   - Best VQA performance  
   - Lowest hallucination rates  

3. **MedGemma considerations**:  
   - Specialized for medical accuracy (best in medical hallucination test)  
   - Limited functionality (no bounding box support)  
