# TimesFM: A Decoder-only Foundation Model for Time-Series Forecasting

**Paper:** *TimesFM: A Decoder-only Foundation Model for Time-Series Forecasting*  
**Conference:** ICML 2024

---

## 🧩 Introduction

- The goal is to build a **large pretrained foundation model** for **time series forecasting** with strong **zero-shot performance**.  
- This allows downstream forecasting tasks **without additional fine-tuning** and **reduces computation cost**.

### Key Ideas
1. Train a large-scale model on **real + synthetic time series data**.  
2. Use a **decoder-style attention architecture**.  
3. Achieve competitive or better performance than LLM-based time series models with **smaller model size and dataset**.  

---

## 📚 Related Work

- Earlier deep learning forecasting models often surpassed classical statistical models in specific datasets.  
- However, no prior work had successfully built a **generalizable foundation model** that performs across diverse datasets.  
- Attempts to fine-tune **LLMs** for time series achieved decent zero-shot results, but **TimesFM** outperforms them with a **smaller model and dataset**.

---

## 🧮 Problem Definition

<img width="251" height="39" alt="image (3)" src="https://github.com/user-attachments/assets/2489c9fa-4628-4773-9b00-80dfe9924c01" />

Given a time series \( y(1:L) \), the model predicts \( y(L+1:L+H) \) for **point forecasting** (minimizing MAE).  
- **L:** lookback length  
- **H:** forecast horizon  

---

## 🧠 Model Architecture

A robust foundation model should handle **arbitrary context** and **variable horizon lengths**.  
TimesFM introduces several architectural choices to achieve this.

### 🔹 Key Design Elements

1. **Patching**  
   - Inspired by patch-based models, the time series is divided into **non-overlapping patches**, improving both accuracy and inference speed.  

2. **Decoder-only model**  
   - Sequentially predicts the next patch given previous ones, similar to autoregressive text decoders.  

3. **Longer output patches**  
   - Instead of predicting one sample at a time, the model predicts **longer output patches**, which improves efficiency and performance.  

4. **Patch masking**  
   - To prevent overfitting to specific multiples of input patch lengths, **random masking** is applied to starting positions, improving generalization across sequence lengths.  

<img width="827" height="502" alt="image (4)" src="https://github.com/user-attachments/assets/c9ce552c-3df0-43a2-9149-dbf5b943d342" />

---

### 🔹 Input Layers

- The time series is divided into **non-overlapping patches**.  
- Each patch is passed through a **residual block** for embedding.  
- Positional encodings are added to preserve order and temporal context.

---

### 🔹 Stacked Transformer

- The core is a **stacked Transformer block** using **causal attention**,  
  meaning each output token attends only to current and past input tokens.

---

### 🔹 Output Layers

- Trained in a **decoder-only** fashion.  
- Each output token is predicted based on all previous input patches.  
- Unlike many models, **input patch length and output patch length can differ**.  
- Output tokens go through a residual block to produce forecasts for the output patch length.  

---

### 🔹 Loss Function

- Uses **MSE loss** for point forecasting.  
- Can be extended to **probabilistic forecasting** by minimizing a **maximum likelihood loss**.

---

### 🔹 Masking Strategy

- During training, sample a random number \( r \in [0, p-1] \) where \( p \) is the input patch length.  
- Replace the first \( r \) samples of the input sequence with zeros.  
→ Helps the model generalize to random input sequence lengths.  

---

## 🧠 Pretraining Details

**Training data sources:**
1. **Google Trends** – search data (hourly, daily, weekly, monthly over 15 years)  
2. **Wikipedia Pageviews** – hourly view counts  
3. **Real-world datasets:** Electricity, Traffic, Weather (various frequencies, hourly–15min)  
4. **Synthetic datasets:** sinusoidal, trend-based, and other generated signals to capture missing granularities  

<img width="709" height="588" alt="image (5)" src="https://github.com/user-attachments/assets/2536c04c-3d44-4e34-a45a-7501b156a7a6" />

---

## 📈 Results

### Zero-Shot Evaluation

<img width="1135" height="284" alt="image (6)" src="https://github.com/user-attachments/assets/cc3473df-4f6b-4d6d-937c-fb9a7833dd91" />

- Uses **Scaled MAE** to normalize across datasets with different scales.  
- Even without training on **Monash** data, **TimesFM** performs competitively.  
- Achieves strong results on **Darts** benchmarks (note: `llmtime` may have data leakage).  
- On **ETT** (long-horizon forecasting benchmark), TimesFM achieves the **best accuracy** and **significantly lower computation time** than alternatives.

---

## 🔬 Ablation Studies

<img width="981" height="719" alt="image (7)" src="https://github.com/user-attachments/assets/2f60d25a-802a-4701-99ce-c21ece9478f8" />

1. **Scaling:**  
   - Tested models with **17M**, **70M**, and **200M parameters**.  
   - Errors decrease proportionally with parameter size.  

2. **Autoregressive Decoding:**  
   - Predicting the entire horizon at once performs better than step-by-step (per-sample) prediction.  
   - Longer output patches yield improved results.  

3. **Input Patch Length:**  
   - Very large input patches make the model act like an encoder, reducing decoder advantages.  
   - Best results with **input patch length = 16 or 32**; 32 offers the fastest inference time.  

4. **Synthetic Data Contribution:**  
   - Improves performance on datasets like **Monash** and **ETTm**, confirming synthetic data helps learn finer granularity.  

---

## 🧭 Conclusion

- **TimesFM** is a **zero-shot forecasting foundation model** trained on diverse **real and synthetic datasets**.  
- It achieves **near-supervised performance** while remaining efficient and compact.  
- Demonstrates robust generalization across different temporal resolutions.

---

## ⚠️ Limitations & Future Work

- **Prompt Tuning:** Explore prompt-based conditioning similar to LLMs for time series.  
- **Probabilistic Forecasting:** Extend beyond point predictions to model uncertainty.  
- **Covariate Handling:** Current model ignores exogenous variables.  
- **Fine-tuning Studies:** Investigate methods to incorporate covariates during finetuning.  
- **Architectural Exploration:** Hyperparameter tuning was limited — alternative designs could improve results.  
- **Interpretability:** Like many large deep models, TimesFM remains less interpretable due to data scale and architectural depth.
