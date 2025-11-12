# MOMENT: A Family of Open Time-series Foundation Models

**Paper:** *MOMENT: A Family of Open Time-series Foundation Models*  
**Published:** ICML 2024

---

## 📘 Introduction

- Due to the **practical value of time series tasks**, a foundation model for time series is proposed.  
- The model aims to handle multiple tasks: **forecasting, anomaly detection, classification, and imputation**.  
- **MOMENT** achieves strong **zero-shot performance** and supports **fine-tuning**.  
- Introduces a large-scale pretraining dataset called **Time Series Pile**.

### Key Contributions

- Proposes **Time Series Pile**, a large and diverse pretraining dataset.  
- Performs **large-scale mixed-data pretraining**.  
- Evaluates the model on **five distinct tasks**.  
- Confirms that **time series characteristics** (frequency and trend) are captured within representations.  
- Demonstrates **cross-modal transfer**, showing that a time-series-trained model can achieve reasonable performance on vision classification tasks.

---

## 📚 Related Work

- **Transformer and patching:**  
  Efficient and effective structure achieved by applying patching, similar to vision transformers.

- **Masked representation learning:**  
  Learns to reconstruct masked regions, inspired by masked language and image modeling.  
  → Enables **self-supervised learning** from unlabeled data.

- **Contrastive representation learning:**  
  Relies on data augmentation; some works learn to reconstruct after zero-masking.  
  → Masked reconstruction learning is found to be more effective for **forecasting and imputation** tasks.

---

## ⚙️ Methodology

### Time Series Pile

A collection of datasets spanning four task types:

1. **Informer datasets** – Long-horizon forecasting (ETT, Electricity, Traffic, etc.; 9 datasets)  
2. **Monash archive** – Forecasting benchmark with 58 short-horizon datasets  
3. **UCR/UEA archive** – Time-series classification datasets (includes ECG signals)  
4. **TSB-UAD benchmark** – Anomaly detection (1,980 univariate time series)

- Used the **original train/test splits** defined by dataset authors.  
- When not provided:
  - Long-horizon datasets → horizontally split  
  - Short-horizon datasets → vertically split  


<img width="631" height="340" alt="image (3)" src="https://github.com/user-attachments/assets/4aa53a84-273d-4588-91de-3f2c3ecbceed" />

---

## 🧩 Model Architecture

<img width="606" height="340" alt="image (4)" src="https://github.com/user-attachments/assets/4652476d-2c24-4521-8208-4524a432a98c" />


- Time series inputs are **patched** into smaller segments.  
- Random patches are **masked**, and the model learns to reconstruct them.  
- The model takes **univariate time series** as input and applies an equal-length **binary mask** (1, T).  
- Before patching, **instance normalization** is applied to each sample.

### Random Masking
- Masked patches are replaced with a **learnable [MASK] embedding**.  
- Normal patches go through standard patch embedding.  
- All masked patches share the same learnable embedding vector.

### Handling Missing Values
- Patches containing missing values are also replaced with the **[MASK] embedding**.  
- The transformer encodes all patch embeddings, and reconstruction is applied to **all patches**.  
- Training loss = **MSE between reconstructed and ground-truth patches** (masked patches only).

---

### Model Design Components

**Handling variable-length series**
- Input length fixed to **512**.  
- Longer series are subsampled; shorter ones are zero-padded.  
- **Instance normalization** ensures local stationarity.  
- Sampling rates are not used (some tasks lack this metadata).

**Lightweight prediction head**
- Used for task-specific fine-tuning.  
- Most parameters reside in the encoder; the head performs final projection based on task type.

**Positional Embedding**
- Uses **relative positional embedding**, but also adds **absolute sinusoidal embeddings** to each patch.

---

## 🧠 Pretraining

- During training, a subset of patches is randomly replaced by learnable **[MASK] embeddings**.  
- Masked patches are reconstructed to minimize **masked reconstruction error (MSE)**.  
- Objective: reconstruct masked patches, not the entire sequence.

### Pretraining Setup

- Follows the size scaling of **T5 models** for encoder depth and width:
  | Model | Layers | Embedding Dim | Parameters |
  |--------|---------|---------------|-------------|
  | Small | 6 | 512 | 40M |
  | Base | 12 | 768 | 125M |
  | Large | 24 | 1024 | 385M |

- Sequence length: 512  
- Patch size: 8 → 64 patches per sequence  
- Optimizer: **Adam**  
- Scheduler: **Cosine learning rate**  
- Trained for **2 epochs**

---

## 🧩 Fine-Tuning on Downstream Tasks

### Forecasting
- The **reconstruction head** is replaced by a **forecasting head**.  
- Flattened patch embeddings (N × D) are projected into horizon-length predictions (H).  
→ Converts entire sequence embeddings into future projections.

---

### Experiment Setup

- Compared primarily with **TimesNet**, a general-purpose time series model.  
- Evaluated on **five tasks** under limited compute and supervision resources.  

**Metrics:**
- **MSE**, **MAE** for long-horizon forecasting  
- **sMAPE** for short-horizon forecasting  
- **Adjusted F1-score** for anomaly detection  

**Baselines:**
- State-of-the-art models for each task  
- MOMENT (large variant, batch size 64)

---

## 📈 Results

### Long-Horizon Forecasting
- Near SOTA performance on most datasets  
- Outperforms some large language model–based baselines  

<img width="1055" height="700" alt="image (5)" src="https://github.com/user-attachments/assets/5fa088b2-bad6-479d-8b78-9997401962d0" />

- **PatchTST** shows highest accuracy overall.  
- **N-BEATS** sometimes exceeds recent transformer models.

---

### Short-Horizon Forecasting

<img width="1059" height="281" alt="image (6)" src="https://github.com/user-attachments/assets/9a7e7940-96d2-4626-9316-3694b3c10fd7" />

- Still room for improvement  
- Statistical methods perform strongly, though MOMENT outperforms them on several datasets

---

### Classification

<img width="511" height="299" alt="image (7)" src="https://github.com/user-attachments/assets/200397ad-08db-4033-89f9-19ac3e6cf53d" />

- Without task-specific fine-tuning, embeddings trained with SVM surpass other classification models.  
- Visualizations show **clear class clustering** in 2D embedding space.

---

### Anomaly Detection

<img width="1068" height="434" alt="image (8)" src="https://github.com/user-attachments/assets/78531270-1499-4f21-8de2-5e8126106c61" />

- MOMENT achieves **SOTA performance** in most metrics.  
- KNN shows slightly higher VUS ROC, but MOMENT dominates overall.

---

### Imputation

<img width="1073" height="278" alt="image (9)" src="https://github.com/user-attachments/assets/38158211-16a2-40f7-bbb5-1f613a863a0e" />

- Shows **lowest reconstruction error** on the ETT dataset.  
- Learns **trend, amplitude, frequency, and phase** components effectively.  
- Struggles to distinguish vertically shifted signals (due to normalization).  
- Classification results confirm strong **zero-shot representations**.

---

## 🧭 Properties of Large Time Series Models

<img width="265" height="200" alt="image (10)" src="https://github.com/user-attachments/assets/d25b68aa-a678-42c2-b6b7-57867df9192b" />

- Training loss decreases with increasing parameter count.  
- Models initialized **from scratch** outperform those initialized with **Flan-T5** weights.  

### Cross-Modal Sequence Learning

- Large pretrained language and vision transformers also generalize to sequence learning tasks.  
- MOMENT demonstrates potential to transfer from time series to **image or text classification** tasks.

<img width="508" height="259" alt="image (11)" src="https://github.com/user-attachments/assets/95ea517c-4219-432b-a16d-03af8c3eb312" />

- When attention and FFN layers are frozen, MOMENT achieves comparable performance to GPT-2 and FLAN-T5.  
- With sufficient data, models trained **from scratch** outperform those initialized with language model weights → indicating that **Time Series Pile** provides enough data for full pretraining.

---

## 🧩 Conclusion

- Constructed a large-scale dataset: **Time Series Pile**.  
- Tested three model sizes; performance scales with parameter count.  
- Achieved strong results across multiple time series tasks — near or at SOTA on **anomaly detection** and **classification**.  
- Short-horizon forecasting remains the most challenging area.  

### Future Work

- Extend to **cross-modal and multimodal** learning directions.  
- Improve **forecasting performance** using causal attention or task-specific modifications.
