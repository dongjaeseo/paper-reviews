# MANTIS: Lightweight Calibrated Foundation Model for User-Friendly Time Series Classification

**Paper:** *MANTIS: Lightweight Calibrated Foundation Model for User-Friendly Time Series Classification*  

---

## 🧩 Summary

- State-of-the-art (SOTA) **Time Series Foundation Model (TSFM)** for classification  
- Uses **contrastive learning (CL)** with a **Vision Transformer (ViT)**–style architecture (uses CLS token)  
→ Patches time series, applies Transformer, uses CLS embedding to represent the sequence, then applies CL  
- Employs **adapter modules** to handle **multivariate inputs** (tested on datasets with 1000+ channels)

---

## 📘 Introduction

- A single general-purpose model that performs multiple time series tasks often shows **suboptimal performance**  
  → Each task benefits from a different loss function  
    - e.g., **Imputation**: masked reconstruction loss  
    - **Classification**: contrastive loss  

- Among TSFMs, models specifically designed for **classification** are rare  
  → This gap motivates the development of **MANTIS**

---

## ⚙️ Methodology

### Overall Workflow

1. Split a time series into **patches**  
2. Learn a **global representation** of the sequence via a **CLS token**  
3. **Pretraining:** contrastive learning on unlabeled datasets (X)  
4. **Finetuning:** supervised learning using labeled datasets (X, Y) with a classification head  

---

### Handling Multivariate Inputs

- The model is **originally trained on univariate series**  
- For multichannel data, each channel is treated as an **independent univariate series**  
- Extract embeddings from each channel and **concatenate**  
- When the dimensionality is large, reduce it from *d* → *d_new* via an **adapter module**  

---

### Preprocessing

- Since time series vary in sequence length and sampling rate, preprocessing is required for both pretraining and inference.  
- Like resizing images to a fixed input size (e.g., 224×224 in CV),  
  → time series are **resized to a fixed length (512)** before processing.  
- Uses **instance-level standard scaling** to normalize each sample by its mean and variance — handling datasets with different sampling rates.

---

## 🧠 Architecture

Applies the **Vision Transformer (ViT)** architecture to time series data.

<img width="1094" height="468" alt="image (3)" src="https://github.com/user-attachments/assets/4b1dccfb-0297-4667-9aeb-c657c945f10e" />

---

### 🔹 Token Generator Unit

- Performs **instance-level normalization**  
- Patches the series and applies **1D convolution** to split it into 32 patches  
  → A 1×T signal is passed through convolution and mean pooling → reshaped to (256, 32)

- Introduces **time series differential patches** to address **non-stationarity** by computing differences between adjacent timestamps — more stable across samples  
- Since instance normalization removes measurement scales, the model computes and encodes **patch-level mean and standard deviation** via a *Multi-Scaled Scalar Encoder*  
- Concatenates all features (original TS, differential, and statistics) and projects them to **32 tokens × 256 dimensions**

---

### 🔹 ViT Unit

- Adds a **CLS token** (global token) to the 32 input tokens — a learnable vector used to represent the entire sequence  
- Uses **sinusoidal positional encoding** added to the input tokens  
- 6 Transformer layers with **8-head multi-head attention**  
- The **CLS token** serves as the sequence-level representation  

---

### 🔹 Projector

- **Pretraining:** uses a linear layer for projection (contrastive learning objective)  
- **Finetuning:** replaces projection with a **classification head** for labeled tasks  

---

## 🧪 Pretraining

- Uses **contrastive learning**  
- Two augmented versions of the same time series are created, and their embeddings are trained to be close via cosine similarity  

### Data Augmentation

<img width="385" height="400" alt="image (4)" src="https://github.com/user-attachments/assets/6268ca49-81d3-4925-9e05-6a4854d8afdd" />

- Many augmentation functions were tested  
- Since heavy distortions may destroy key temporal patterns, they adopted **random crop and resize**  
- Crop ratio kept small (0–20%) to preserve critical information  

Some pretraining data included **sleep-related datasets** (ECG, EMG, Epilepsy, Sleep-EEG from PhysioNet).

---

## 🔧 Adapter

- Addresses the challenge of handling **multivariate data**  
- While MANTIS can be applied independently to each channel (univariate mode), this is inefficient and lacks inter-channel correlation  
- Introduces **adapter modules** to reduce and mix channel dimensions:  
  - Transforms *d* channels → *d_new* channels  
  - Each transformed channel is fed through the model, and embeddings are concatenated afterward  

**Five adapter types tested:**
1. PCA  
2. SVD  
3. Random projection  
4. Variance-based channel selection (VarSelector)  
5. Differentiable linear combiner  

---

## ⚗️ Experiments

### Baselines
- **UniTS**  
- **GPT4TS**  
- **NuTime**  
- **MOMENT**

### Datasets (Evaluation)
- **UCR** – 128 univariate datasets  
- **UEA** – 27 multivariate datasets  
- **Blink**, **MotionSenseHAR**, **EMOPain**, **SharePriceIncrease**

---

### Experimental Settings
- **Zero-shot evaluation:** freeze encoder, extract embeddings, train Random Forest classifier  
- **Fine-tuning:** supervised training with hyperparameter tuning  
- **Ablation, Adapter, Calibration** experiments conducted  

---

### 🔸 Zero-Shot Results
<img width="303" height="314" alt="image" src="https://github.com/user-attachments/assets/245aa278-d368-414a-be64-d41b5cd55e72" />

Shows the **most robust overall performance** among time series classification models.  

---

### 🔸 Ablation Study

**Architectural results:** adding differential time series improves accuracy.  

**Finetuning strategy comparison:**  
| Setting | Description |
|----------|--------------|
| RF | Frozen encoder + Random Forest classifier |
| Head | Frozen encoder + linear classification head |
| Scratch | Randomly initialized encoder + classification head |
| Full | Pretrained encoder + classification head (fine-tuned) |

**Metrics:** test accuracy at last epoch and best accuracy across epochs.  

**Findings:**
- **Full vs Scratch:** pretraining is effective  
- **Full vs Head:** full fine-tuning yields best results  
- **RF vs Linear:** RF slightly higher  
- Difference between best and last epoch small → minimal need for heavy hyperparameter tuning  

<img width="754" height="313" alt="image" src="https://github.com/user-attachments/assets/bfb4c8d2-c1fa-478b-b2c2-c9e88627e1d4" />


---

### 🔸 Adapter Analysis

- For **classification**, only one class prediction is needed across multiple input channels  
- MANTIS, originally univariate, is adapted to multivariate setups by reducing high-dimensional inputs to 10 key channels, extracting embeddings from each  

**Advantages:**
1. **Channel reduction** → efficient computation  
2. **Channel interaction** → captures dependencies between channels (dataset-dependent)  
3. **Channel selection** → drops uninformative channels; variance-based selector performed best  
4. **Differentiable adapter** → promising but less stable overall; future optimization needed  

No single adapter dominated all datasets, but **VarSelector** achieved the best results (especially for **epilepsy data**).  

<img width="1081" height="639" alt="image (5)" src="https://github.com/user-attachments/assets/e50a4331-160a-463b-b5a4-4ab1a1198a9f" />

---

### 🔸 Calibration

- Calibration ensures **model confidence** aligns with true probabilities  
- Measured using **Expected Calibration Error (ECE)** with 10 bins  

**Methods tested:**
- **Temperature scaling:** adjusts softmax sharpness  
- **Isotonic regression:** step-function-based mapping between logits and labels  

<img width="1073" height="293" alt="image (6)" src="https://github.com/user-attachments/assets/b7f5793b-96fd-4f27-a87c-f821f47b586a" />

---

## 🧭 Conclusion & Future Work

- Lightweight, high-performing **time series foundation model** for classification  
- Strong **calibration** performance compared to baselines  
- Notes that architecture design still has room for improvement —  
  parameter count does not strictly correlate with performance  
- Significant performance gap remains between **fine-tuned** and **zero-shot** settings → zero-shot capability can improve  
- Highlights the need for stronger calibration methods for interpretability  

**Adapters implemented using scikit-learn:**
- **PCA (Principal Component Analysis):**
  - First component = linear combination with highest variance  
  - Each subsequent component captures next-highest variance
- **SVD (Singular Value Decomposition)**
- **Random Projection:**
  - Uses `SparseRandomProjection` from sklearn  
  - Generates random matrix *(n_dim × n_dim_new)* to reduce dimensions  
  - Based on the law of large numbers, spatial relations are approximately preserved after projection  

<img width="577" height="74" alt="image (7)" src="https://github.com/user-attachments/assets/081df297-f41d-484a-93e7-76e62f1ed8af" />
