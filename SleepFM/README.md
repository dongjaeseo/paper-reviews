# SleepFM: Multimodal Foundation Model for PSG Data

---

## 🧠 Research Background

Polysomnography (**PSG**) remains the gold standard for sleep diagnosis, but manual expert scoring is time-consuming and error-prone.  

Most existing deep learning studies:
- Focus on **single-task**, **single-modality**, or **fully supervised** setups  
- Use **pairwise contrastive learning (CL)**, which is limited for multimodal representation learning  

---

## ⚙️ What Makes SleepFM Different

- **First multimodal foundation model** designed for PSG data  
- Incorporates **three modalities**:  
  - **BAS** (Brain Activity Signals) – 10 channels  
  - **ECG** (Electrocardiogram) – 2 channels  
  - **Respiratory signals** – 7 channels  
  → Total of **19 channels**

- Introduces **Leave-One-Out Contrastive Learning (LOO-CL)** as an alternative to pairwise CL  
- Evaluates **LOO vs. Pairwise CL** on several downstream tasks  
- Represents each 30-second PSG epoch as a shared multimodal embedding  
- Performs well even with **limited labeled data** using pretrained feature extractors  

**Downstream tasks studied:**
1. Sleep stage classification  
2. Sleep-disordered breathing (SDB) detection  
3. Demographic prediction (age, gender)  
4. Cross-modal retrieval  

---

## 🔍 Overview

Each modality is processed by its own **1D CNN encoder (EfficientNet-based)**.  
Contrastive learning across modalities allows the model to learn **shared representations** that align information from different physiological signals.

<img width="612" height="769" alt="image (3)" src="https://github.com/user-attachments/assets/b453e5c0-e998-44ee-9993-1e7f5fb356f7" />


---

## 🧩 Model Architecture

**Encoder pipeline**

Conv1d → BatchNorm → Bottleneck(1–2, dropout) → MaxPool →
Bottleneck[2,3,3,3,3] (dropout) → Conv1d →
AdaptiveAvgPool → ReLU → Dropout → Fully Connected

## 📊 Data Setup

PSG signals are segmented into **30-second epochs**, sampled at **256 Hz** → **7,680 data points per epoch**.

| Modality | Input Shape |
|-----------|-------------|
| BAS | (10, 7680) |
| ECG | (2, 7680) |
| Respiratory | (7, 7680) |

Each encoder outputs a **[batch, 1280]** embedding.  
Contrastive objectives are applied over these embeddings.

---

## 🔁 Contrastive Learning (CL)

**Objective:** increase similarity between positive pairs and decrease it for negatives.  

---

### Pairwise CL

- Encourages embeddings from two modalities recorded at the same time to be similar  
- Computes loss only for positive pairs  
- Temperature parameter adjusts softmax sharpness  

**Training process:**
1. Compute cosine-similarity matrix `[batch × batch]`  
2. Diagonal entries correspond to matching timestamps  
3. Maximize diagonal similarity vs. others  
4. Combine symmetric losses (i→j, j→i)  
<img width="548" height="90" alt="image (4)" src="https://github.com/user-attachments/assets/938de5bd-6d57-4c9c-b98b-e68b5ba0da06" />



---

### Leave-One-Out CL (LOO-CL)

- Each modality embedding is compared to the **mean embedding of all other modalities** at the same timestamp  
- For sample *k*, the loss for modality *i* uses the averaged embedding of modalities ≠ *i*  
<img width="557" height="82" alt="image (5)" src="https://github.com/user-attachments/assets/37bbb165-62f9-4f64-8e3c-af94ee9180e5" />



---

## 🧪 Experiments

Data were divided into **pretrain**, **train**, **validation**, and **test** splits.  
Encoders were pretrained with CL and later used to generate embeddings for downstream tasks.  

**Baseline:** a supervised CNN trained without CL.

---

### 1️⃣ Demographic Attribute Classification

**Goal:** verify embedding quality  
**Method:** logistic regression on SleepFM embeddings (age/gender)  
**Metrics:** AUROC / AUPRC  

**Results:**  
- **LOO-CL** achieved the best results  
- **Pairwise CL** exceeded CNN baseline  
→ CL embeddings captured demographic patterns effectively  

<img width="1200" height="279" alt="image (6)" src="https://github.com/user-attachments/assets/92231b19-a8ef-4082-819b-654edc3930c7" />
<img width="544" height="161" alt="image (7)" src="https://github.com/user-attachments/assets/a1286589-a206-4dda-86ba-df23f3169b4a" />


---

### 2️⃣ Cross-Modal Retrieval

**Goal:** test whether embeddings from one modality can retrieve corresponding ones in another  

**Metrics**
- **Recall@10:** proportion of correct matches within top 10 (higher = better)  
- **Median Rank:** median position of true pair (lower = better)  

**Results**
- **LOO-CL:** 500–8000× higher Recall@10 vs. random retrieval
- <img width="554" height="200" alt="image (8)" src="https://github.com/user-attachments/assets/09b161de-c051-4d74-a3e4-7101a63b2174" />

- **Pairwise CL:** performed slightly better here, as retrieval aligns with its objective  
<img width="553" height="202" alt="image (9)" src="https://github.com/user-attachments/assets/a82132c4-cf3b-4321-8a9a-17b7dd2a732b" />



---

### 3️⃣ Downstream Classification

**Tasks**
- Sleep-stage classification (5 classes)  
- SDB detection (binary)

**Findings**
- **LOO > Pairwise > CNN**
→ LOO CL offered the most generalizable representations  
<img width="1273" height="314" alt="image (10)" src="https://github.com/user-attachments/assets/072a89c6-e942-493c-bfc5-6a8ed4ccbcae" />

<img width="536" height="160" alt="image (11)" src="https://github.com/user-attachments/assets/9e93878b-aeb7-47f4-9fa1-43bd9525f9e8" />


---

### 4️⃣ Few-Shot Evaluation

**Goal:** measure robustness with limited data  
**Method:** vary training samples (k = 1, 2, 4 … 1265)  

**Outcome:**  
LOO-CL consistently showed the highest AUROC/AUPRC → strong few-shot performance  

<img width="1258" height="321" alt="image (12)" src="https://github.com/user-attachments/assets/11adff03-72c4-4e55-848d-35a70694054d" />


---

## 🧪 Ablation Studies

Trained CL models with different modality counts (3 / 2 / 1).  

**Evaluated on**
- BAS → sleep-stage classification  
- Respiratory → SDB classification  

**Observations**
- 3-modality pretraining gave best overall results  
- ECG contributed strongly (BAS-ECG and Resp-ECG performed well)  
- BAS-Respiratory without ECG lagged behind  
→ modality choice significantly impacts downstream utility  
- Single-modal models performed poorest  

<img width="1241" height="315" alt="image (13)" src="https://github.com/user-attachments/assets/1f3042fc-ab5f-4173-bd3d-c16ff28f3e73" />


**External dataset (PhysioNet):**
- SleepFM outperformed CNNs trained on external data, despite no fine-tuning  
- Maintained strong results despite EEG configuration differences  
<img width="1019" height="314" alt="image (14)" src="https://github.com/user-attachments/assets/0a44bf9a-43a4-4654-8a1b-7aef1f51784f" />



---

## 🧭 Conclusion

- **SleepFM** performs well across multiple tasks — demographic prediction, retrieval, classification — using **multimodal contrastive learning**  
- **LOO-CL** effectively handles multimodal inputs and outperforms pairwise CL in representation learning  
- **Pairwise CL** remains slightly better for retrieval tasks  
- The model generalizes robustly to unseen datasets, suggesting potential for **clinical deployment** even with limited labels  

## ⚠️ Limitations

- Dataset limited to one institution  
- Other sleep-related tasks (arousal, limb movements, narcolepsy) not explored  
- Extending CL toward broader self-supervised frameworks remains future work  
