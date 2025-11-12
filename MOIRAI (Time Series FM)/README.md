# MOIRAI: Unified Training of Universal Time Series Forecasting Transformers

**Paper:** *MOIRAI: Unified Training of Universal Time Series Forecasting Transformers*  
**Published:** ICML 2024

---

## 🧩 Introduction

This paper addresses several key challenges in **time series forecasting**:

<img width="652" height="395" alt="image (3)" src="https://github.com/user-attachments/assets/a5d70b4b-af4d-4253-bdf4-c731c85cf826" />

1. **Frequency diversity** – Models suffer from negative interference when training on data with different sampling rates (minutely, hourly, daily, etc.).  
2. **Multivariate forecasting** – Lack of foundation models capable of forecasting multiple correlated variables.  
3. **Probabilistic forecasting** – Different datasets follow different statistical distributions (e.g., symmetric, skewed, count-based).  
4. **Lack of large, diverse datasets** – Limited data for large-scale pretraining in time series domains.

---

### 🔹 Solutions Proposed by MOIRAI

- **Frequency Handling:**  
  Uses *multiple input and output projection layers* to learn from time series with varying frequencies.  
  Patch sizes are adjusted dynamically to capture patterns at multiple temporal resolutions.

- **Multivariate Handling:**  
  Introduces the **Any-Variate Attention** mechanism to manage arbitrary numbers of variables.  
  Time and variable dimensions are flattened into a single sequence:

   ([var0_time0], [var0_time1], … [var1_time0], …)

  Two key techniques enable this:
1. **Rotary Position Embedding (RoPE)** – embeds temporal indices.  
2. **Learned Binary Attention Biases** – help the model differentiate between variables after flattening.

- **Probabilistic Forecasting:**  
Employs a **mixture of parametric distributions** to flexibly model different predictive distributions (count, skewed, symmetric, etc.).

- **Large-Scale Dataset:**  
Introduces **LOTSA**, a dataset with **27 billion observations** across **9 domains**, specifically designed for training foundation models.

<img width="1269" height="357" alt="image (4)" src="https://github.com/user-attachments/assets/b0be8312-0e2a-44e4-af53-54ab3c483660" />

---

## 📚 Related Work

### Pretrained Forecasting Models

<img width="623" height="259" alt="image (5)" src="https://github.com/user-attachments/assets/c994bbeb-fe42-47e3-b54c-be466d43d9b5" />

- Comparison of existing pretrained models against the challenges above.

### Pretrain + Finetune Paradigm

- Prior research focused on **pretraining and finetuning within a single dataset**.  
- **MOIRAI** demonstrates stronger **cross-domain generalization**.  
- Builds upon ideas from **denoising autoencoders** and **contrastive learning**.

---

## ⚙️ Method

### Problem Formulation

- Predict multivariate series **Y(i)** (targets) given covariates **Z(i)** over horizon **h**, using past window length **l**.

<img width="274" height="37" alt="image (6)" src="https://github.com/user-attachments/assets/726500bd-caf4-411d-9649-61abde6bcb5a" />

- Unlike point forecasting, MOIRAI performs **probabilistic forecasting** — adding uncertainty estimation.

<img width="133" height="39" alt="image (7)" src="https://github.com/user-attachments/assets/1a4a985d-0290-42a6-aa82-3c8c5ca9bdb7" />

- The model learns distribution parameters ϕ for each horizon h by maximizing log-likelihood with respect to the ground truth.

<img width="390" height="73" alt="image (8)" src="https://github.com/user-attachments/assets/996cbc61-7ac8-4db8-806f-0a0c7ba80308" />

---

## 🧠 Architecture

<img width="1282" height="623" alt="image (9)" src="https://github.com/user-attachments/assets/3b647eed-bb3f-450f-89e5-a8ee9605e6dd" />

- MOIRAI adopts a **masked encoder** architecture with patch-based inputs.  
- Multivariate time series are **flattened** into a single sequence across both time and variable dimensions.  
- Patches are embedded into vectors and processed by a transformer encoder.

### Multi–Patch-Size Projection Layer

- Adjusts patch size dynamically to handle diverse temporal frequencies.  
- Forecast horizons are masked with a learnable mask embedding to teach the model to reconstruct missing patches.  
- The transformer output passes through a **multi–patch-size projection layer**, matching the input patch size to produce probabilistic forecasts.

---

### Optimization Enhancements

- Inputs and outputs are **instance-normalized** to accommodate varying frequencies.  
- Optimization strategies inspired by LLMs:
1. Replace **LayerNorm** with **RMSNorm**  
2. Apply **Query–Key normalization**  
3. Use **SwiGLU** in feed-forward layers  
4. Remove biases in transformer layers  

---

### Any-Variate Attention

- Enables the model to forecast an arbitrary number of variables.  
- Time and variable dimensions are flattened into one sequence, with **variate encodings** distinguishing each variable (analogous to positional encoding).

<img width="477" height="93" alt="image (10)" src="https://github.com/user-attachments/assets/eb99c0ab-6d34-4d90-aaba-f17a8f36b5c6" />

- **Attention score** between variable m at time i and variable n at time j:  
incorporates a **rotary matrix (R)** to model temporal distance and **binary attention biases** to separate variable effects.

---

### Mixture Distribution

- Uses a **mixture of parametric distributions** for flexible probabilistic forecasting while keeping sampling and loss simple.

<img width="395" height="88" alt="image (11)" src="https://github.com/user-attachments/assets/17c90c4f-7d86-4d65-88b1-0fdfa0f0a2ca" />

**Model Output:**  
\( ϕ = {w_1, ϕ_1, …, w_4, ϕ_4} \)  
- \( ϕ_i \): parameters of the i-th probability distribution (mean, variance, shape, etc.)  
- \( w_i \): mixture weights  

**Supported Distributions:**
1. *Student’s t* — general-purpose continuous data  
2. *Negative Binomial* — count data (e.g., web traffic)  
3. *Log-normal* — skewed economic or natural phenomena  
4. *Low-variance Normal* — high-confidence signals  

---

## 🏋️ Training

### Pretraining Objectives

To optimize the mixture distribution, MOIRAI uses **two sampling strategies**:

**1. Data Distribution Sampling**
- LOTSA contains diverse domains and frequencies.  
- Time series samples are drawn from each sub-dataset to balance domain bias.

**2. Task Distribution Sampling**
- Randomly sample lookback and horizon lengths to train over varying sequence lengths (2–512).  
- Forecasting horizon sampled within [0.15, 0.5] of total series length.  
- For multivariate data:
- Randomly select a subset of variables.  
- Construct pseudo-multivariate series from univariate data.

**Model Sizes:**  
- Small (14M), Base (91M), Large (311M) parameters.

---

## 🧪 Experiments

### In-Distribution Forecasting

<img width="621" height="352" alt="image (12)" src="https://github.com/user-attachments/assets/48eeec6f-d889-4268-84c7-25b105996f10" />

- Evaluated on **Monash** datasets (part of LOTSA).  
- Competes strongly both in-domain and across domains.

---

### Out-of-Distribution Forecasting

#### 1️⃣ Probabilistic Forecasting

<img width="964" height="297" alt="image (13)" src="https://github.com/user-attachments/assets/f2203e69-6b91-4096-9813-b9a8b8b79c57" />

- Compared against **SOTA full-shot models** (since most foundation models are closed-source).  
- Metrics:
- **CRPS** (Continuous Ranked Probability Score): measures forecast–GT distance  
- **MSIS** (Mean Scaled Interval Score): evaluates prediction interval quality (lower = better)  
- **MOIRAI (Base & Large)** achieves **best or second-best performance** across most datasets.

---

#### 2️⃣ Long-Sequence Forecasting

<img width="964" height="299" alt="image (14)" src="https://github.com/user-attachments/assets/b2d3e216-553f-43b7-b18f-f1341d4cb729" />

- Uses the **median** of the predicted distribution as a point forecast.  
- Large model shows minor instability, discussed under limitations.

---

## 🔬 Ablation Study

<img width="465" height="229" alt="image (15)" src="https://github.com/user-attachments/assets/75ab21e2-0969-4d31-a6c8-82776f8caaa1" />

1. **Multi–Patch-Size Component:**  
 - Random or fixed patch sizes reduce performance → multi-size learning is crucial.  
2. **Any-Variate Attention:**  
 - Removing it significantly degrades performance.  
3. **Mixture Distribution:**  
 - Fixing to a single *Student’s t* distribution leads to symmetric, less expressive predictions.

<img width="664" height="387" alt="image (16)" src="https://github.com/user-attachments/assets/30faef95-ab61-4444-b43f-1562e8fa09b9" />

- **Packed Training** improves performance by increasing effective observations per batch without more computation.  
(Adopted from LLM training — replaces padding with shorter parallel sequences.)

---

## 📊 Further Analysis

### Context Length

<img width="549" height="237" alt="image (17)" src="https://github.com/user-attachments/assets/6b8c5df2-0ff6-4ef8-8a99-31d287ad2f47" />

- Unlike previous Transformer forecasters (whose performance plateaued with longer contexts),  
**MOIRAI benefits consistently from extended context windows.**

---

### Packing Efficiency

<img width="551" height="264" alt="image (18)" src="https://github.com/user-attachments/assets/5d2cca41-9c46-47e2-90f8-c1011cbc9cb3" />

- Introduces **packing**, a technique from LLMs, rarely used in time series.  
- Reduces computational overhead from padded sequences.  
- Packing improves performance by **~16%**, as over **60% of sequences** in LOTSA are shorter than the maximum length.

---

## 🧭 Conclusion

- **MOIRAI** is a **foundation model** that tackles major challenges in time series forecasting.  
- Trained on **LOTSA**, an open large-scale time series dataset spanning multiple domains.  
- Performs strongly in both **in-distribution** and **out-of-distribution** settings, achieving **competitive or superior zero-shot results** for probabilistic and long-sequence forecasting tasks.

---

## ⚠️ Limitations & Future Work

- Limited hyperparameter tuning due to computational constraints.  
- Multi–patch-size cross-frequency design is experimental — could be made more flexible.  
- Struggles with extremely high-dimensional multivariate series.  
- Potential improvements in **mask reconstruction** via *latent diffusion*.  
- Future directions: **multimodal extensions** combining time series with tabular or text inputs.

