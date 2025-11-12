# 💤 SleepFM: Multimodal Foundation Model for PSG Data

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

![image.png](attachment:77421513-1655-442e-994f-d4c98d4765bb:image.png)

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

![Pairwise CL](path/to/pairwise.png)

---

### Leave-One-Out CL (LOO-CL)

- Each modality embedding is compared to the **mean embedding of all other modalities** at the same timestamp  
- For sample *k*, the loss for modality *i* uses the averaged embedding of modalities ≠ *i*  

![LOO CL](path/to/loo.png)

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

![Demographic Results](path/to/demographic.png)

---

### 2️⃣ Cross-Modal Retrieval

**Goal:** test whether embeddings from one modality can retrieve corresponding ones in another  

**Metrics**
- **Recall@10:** proportion of correct matches within top 10 (higher = better)  
- **Median Rank:** median position of true pair (lower = better)  

**Results**
- **LOO-CL:** 500–8000× higher Recall@10 vs. random retrieval  
- **Pairwise CL:** performed slightly better here, as retrieval aligns with its objective  

![Retrieval LOO](path/to/retrieval_loo.png)  
![Retrieval Pairwise](path/to/retrieval_pairwise.png)

---

### 3️⃣ Downstream Classification

**Tasks**
- Sleep-stage classification (5 classes)  
- SDB detection (binary)

**Findings**
- **LOO > Pairwise > CNN**
→ LOO CL offered the most generalizable representations  

![Sleep Staging](path/to/sleep_stage.png)  
![SDB Classification](path/to/sdb.png)

---

### 4️⃣ Few-Shot Evaluation

**Goal:** measure robustness with limited data  
**Method:** vary training samples (k = 1, 2, 4 … 1265)  

**Outcome:**  
LOO-CL consistently showed the highest AUROC/AUPRC → strong few-shot performance  

![Few Shot](path/to/few_shot.png)

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

![Ablation](path/to/ablation.png)

**External dataset (PhysioNet):**
- SleepFM outperformed CNNs trained on external data, despite no fine-tuning  
- Maintained strong results despite EEG configuration differences  

![External Data](path/to/external.png)

---

## 🧭 Conclusion

- **SleepFM** performs well across multiple tasks — demographic prediction, retrieval, classification — using **multimodal contrastive learning**  
- **LOO-CL** effectively handles multimodal inputs and outperforms pairwise CL in representation learning  
- **Pairwise CL** remains slightly better for retrieval tasks  
- The model generalizes robustly to unseen datasets, suggesting potential for **clinical deployment** even with limited labels  

---

## ⚠️ Limitations

- Dataset limited to one institution  
- Other sleep-related tasks (arousal, limb movements, narcolepsy) not explored  
- Extending CL toward broader self-supervised frameworks remains future work  
