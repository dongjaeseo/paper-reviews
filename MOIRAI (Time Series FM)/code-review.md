# 🔍 MOIRAI Code Review Notes

### Topics
- Types and characteristics of distributions  
- Logic for selecting patch size during training and inference  
- Analysis of distribution output  
- Application of Rotary Position Embedding  

---

## Q. Distributions Attempted in Moirai

**Code directory:** `distribution_{distribution_name}.py`

**Summary:**  
Four distributions are used, each having strengths for different time series patterns.  
(There were also attempts to include Pareto and Laplace distributions.)

---

### 1. Student’s T

**Parameter:** {loc, scale, df}  

<img width="649" height="218" alt="image (3)" src="https://github.com/user-attachments/assets/8ed66836-eddc-48b7-afed-b59f28cb6d02" />

- Similar to a Normal distribution but with a **df (degrees of freedom)** parameter controlling tail length.  
- Performs well on general time series data.  
- Has thicker tails than Normal → captures spikes and outliers better.  
- Since outlier handling is easier than in Normal, Student’s T is used.

---

### 2. Negative Binomial

**Parameter:** {Total count, logit}  

<img width="380" height="200" alt="image (4)" src="https://github.com/user-attachments/assets/daf566ed-7eb7-4213-8e42-a49bd5df9594" />

**Characteristics:**  
- Discrete distribution with different parameters than continuous ones.  
- Custom `log_prob` implemented due to discreteness.  
- Effective for positive integer count data (e.g., patient counts, web visits).

---

### 3. Log Normal

**Parameter:** {loc, scale}  

<img width="643" height="138" alt="image (5)" src="https://github.com/user-attachments/assets/2155d944-24f6-4e5d-bfbe-adebed70957e" />
<img width="326" height="334" alt="image (6)" src="https://github.com/user-attachments/assets/312f08d5-f5cd-4055-8cdb-31fe44023281" />

- Performs well for **right-skewed data** (economic, environmental).  
- Assumes log(variable) follows a Normal distribution.  
- Long right tail → values cluster left, with few large outliers.  
- Common example: **income distribution** (most values low-moderate, few very high).  

<img width="195" height="52" alt="image (7)" src="https://github.com/user-attachments/assets/3ef1e33b-318a-4398-99de-51e13319d38e" />

---

### 4. Normal (Low Variance)

**Parameter:** {loc}  

<img width="627" height="45" alt="image (8)" src="https://github.com/user-attachments/assets/27bd65a9-5c5a-45a2-b46a-c22b1571479d" />

- Continuous probability distribution with `loc` and fixed `scale = 0.001`.  
- Used for **high-confidence predictions**.  
- Sharp-peaked shape, representing strong certainty.  
- The more weight on this component, the more deterministic the prediction.

<img width="159" height="52" alt="image (9)" src="https://github.com/user-attachments/assets/ac59a2c8-729c-41d5-a5c4-9401e839f882" />

---

## Q. Patch Size Selection Logic (Inference)

**Code directory:** `model_moirai_forecast.py` (line 261, `moiraiforecast` module)

<img width="614" height="172" alt="image (10)" src="https://github.com/user-attachments/assets/b8fa53b4-efcf-4dc7-9c94-97b35b2a4733" />
<img width="399" height="75" alt="image (11)" src="https://github.com/user-attachments/assets/3d4c7173-73ce-4235-ad53-96fdcc990041" />

**Summary:**  
All possible patch sizes are tested; the model infers `val_loss` and distribution parameters for each, and only the one with the **lowest val_loss** is selected.

**When `patch_size = 'auto'`:**
- Compute `val_loss` for every candidate patch size.  
- Stack distribution parameters across patch sizes.  
- Choose the parameters from the **lowest val_loss** case.

**Predicted values:**  
In `forecast.py` (line 330), the model samples **100 predictions** from the chosen distribution for visualization.

<img width="463" height="43" alt="image (12)" src="https://github.com/user-attachments/assets/269ca8a0-55c5-447b-b9cf-7b14015883dd" />

---

## Q. Distribution Output Format

**Code directory:** `distribution__base.py`

**Module hierarchy:**  
`MoiraiForecast` → `moiraimodule` → …

| Distribution | Parameters |
| --- | --- |
| Student’s T | df, loc, scale |
| Negative Binomial | total_count, logit |
| Log Normal | loc, scale |
| Normal (Low Var) | loc |

**Summary:**  
The **Mixed Distribution** consists of **weights** and each distribution’s **parameters**.

<img width="1072" height="343" alt="KakaoTalk_20250812_140922917 jpg" src="https://github.com/user-attachments/assets/4e0ac989-29b3-4a26-b2f2-aac36c6c1a21" />

**Main parameters (PyTree format):**
1. `args_dim` – defines the number of projection nodes.  
2. `domain_map` – maps each projection node to its corresponding transformation function.

→ `params_unbounded` → processed by `domain_map` → produces **weights_logit** and **component parameters**.

<img width="1854" height="1207" alt="KakaoTalk_20250812_140922917 jpg (1)" src="https://github.com/user-attachments/assets/b4a4eda3-30e5-4f36-b9a4-0b4122a05dff" />

`weights_logit` is passed through a categorical layer.  
Each component parameter is computed via the corresponding mapping function.

| Domain Map | Function | Distribution Used |
| --- | --- | --- |
| df | 2 + softplus(x) | Student’s T |
| loc | x | Student’s T, Log Normal, Normal |
| scale | softplus(x).clamp_min(epsilon) | Student’s T, Log Normal, Normal |
| total_count | softplus(x) | Negative Binomial |
| logits | x | Negative Binomial |

---

## Q. Mixture Distribution Loss Calculation

<img width="946" height="849" alt="image (13)" src="https://github.com/user-attachments/assets/b5cd1ecd-f40b-43d4-8c8d-4a8759d730ad" />

**Explanation of the final equation:**  
<img width="1440" height="549" alt="image (14)" src="https://github.com/user-attachments/assets/58d9e0f2-a208-470a-a0f3-fb9d2b112755" />

---

## Q. Patch Size Selection Logic (Training)

**Code directory:** `transform_patch.py`

<img width="936" height="455" alt="image (15)" src="https://github.com/user-attachments/assets/00b453b1-728f-455e-a676-c24a28b4da52" />
<img width="837" height="101" alt="image (16)" src="https://github.com/user-attachments/assets/6582aca3-73cd-4f71-9d29-ba17a9e92080" />

**Condition:**  
Heuristic constraints are defined for each sampling rate.

<img width="244" height="332" alt="image (17)" src="https://github.com/user-attachments/assets/f58d222e-d02d-49f6-b943-3da6f3a177ec" />

**Sampling rates (examples):**
- Second  
- Minute  
- Hourly  
- Daily  
- Business Day  
- Weekly  
- Monthly  
- Quarterly  
- Yearly  
- Annually  

---

## Q. Rotary Position Embedding (RoPE)

**Code directory:** `module_position_attn_projection.py`

<img width="901" height="653" alt="image (18)" src="https://github.com/user-attachments/assets/8153b25e-5d29-4e19-a225-12d9b502a212" />

**Explanation of final equation:**  
<img width="1357" height="721" alt="image (19)" src="https://github.com/user-attachments/assets/ef467eea-7873-4781-83f4-b5691b0bec0d" />
