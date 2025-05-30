
# 🧪 LabQAR

**LabQAR: A Manually Curated Dataset for Question Answering on Laboratory Test Reference Ranges and Interpretation**

Laboratory tests play a pivotal role in diagnosing and managing health conditions, with interpretation depending on reference ranges influenced by specimen type (e.g., blood, urine), age, gender, pregnancy, and other factors. Clinical decision support systems must account for this complexity to avoid misdiagnoses.

**LabQAR** (Laboratory Question Answering with Reference Ranges) is a **manually curated dataset** of **multiple-choice questions** built around **550 lab tests**, including annotations on reference ranges, specimen types, and conditions affecting interpretation. We also benchmark the performance of state-of-the-art **large language models (LLMs)**, demonstrating the effectiveness of **GPT-4o** in clinical reasoning tasks.

---

## 📚 Overview

LabQAR is designed to evaluate LLMs on two critical clinical tasks:

1. **Reference Range Prediction** – Predict the correct SI reference range for a given lab test.
2. **Lab Result Classification** – Classify a lab test value as **High**, **Normal**, or **Low**, considering context (specimen, unit, age, gender).

The dataset is provided in **JSON format** and includes two primary sets for model evaluation.

---

## 🧾 Dataset Structure

### 📁 Set 1: Reference Range Prediction

**Task**: Predict the correct lower and upper bound SI reference range values.

```json
[
  {
    "ID": 1,
    "Question": "For the lab test 'Acetaminophen' measuring in 'μmol/L' in Specimen 'Serum, plasma' for 'any gender' and 'any age group', what is the correct lower and upper bound range values in SI reference range?",
    "Answer": "70–200"
  }
]
```

---

### 📁 Set 2: Lab Result Classification

**Task**: Classify a numeric lab result as **High**, **Normal**, or **Low**.

```json
[
  {
    "ID": 1,
    "Question": "For the lab test 'Acetaminophen' measuring in 'μmol/L' in Specimen 'Serum, plasma' for 'any gender' and 'any age group', a value in 'SI reference range' is 341.62. Is the lab test result?",
    "Choices": "A: High\nB: Normal\nC: Low",
    "Answer": "A"
  }
]
```

---

## 🤖 Model Evaluation

Five LLMs were evaluated:

- 🦙 **LLaMA 3.1** (deployed locally via Hugging Face + LangChain)
- 🐊 **GatorTronGPT**
- 🤖 **GPT-3.5**
- 🧠 **GPT-4**
- 💡 **GPT-4o**

**GPT-4o** achieved the best results across:

- Exact match accuracy
- Classification accuracy
- Tolerance-based evaluations (±10%, ±20%)

---

## 🛠️ Environment and Deployment

### ✅ Local LLM (LLaMA 3.1)
- **Source**: Hugging Face
- **Framework**: LangChain (`langchain==0.3.25`)
- **Hardware**: NVIDIA RTX A6000 (48GB VRAM)
- **Inference Settings**: `temperature=0` for deterministic outputs

### 🌐 Remote LLMs
- **Models**: GPT-3.5, GPT-4, GPT-4o (via OpenAI API)
- **Requirements**: Valid API key, proper environment variable setup

---

## 📂 Files Included

| Filename                    | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `set1_reference_range.json` | Reference range prediction questions                                        |
| `set2_classification.json` | Lab value classification questions                                          |
| `annotation_guidelines.pdf`| Instructions followed during data curation                                  |
| `requirements.txt`         | Python dependencies for replication and inference                          |

---

## 📏 Suggested Evaluation Metrics

- ✅ **Exact Match Accuracy** (Set 1)
- 📊 **Classification Accuracy** (Set 2)
- ± **Tolerance-Based Match**: Acceptable predictions within ±10% or ±20% (Set 1)
- 🔁 **F1 Score**, **Confusion Matrix** (Set 2)

---

## 💡 Example Use Cases

- Evaluating LLMs in real-world **clinical decision-making**
- Fine-tuning models for **lab result interpretation**
- Building **AI diagnostic agents** with LangChain or RAG architectures

---


## 📘 Citation

Please cite this dataset as:

> Bhasuran, B. et al. *LabQAR: A Manually Curated Dataset for Question Answering on Laboratory Test Reference Ranges and Interpretation*. 2025.

---

## ⭐ Acknowledgments

This work integrates expert medical resources including Laposata's Laboratory Medicine, Stanford Medicine, and others for high-quality clinical curation.

---

## 📄 License

This project is made available under the [MIT License](LICENSE).
