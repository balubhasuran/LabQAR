
LabQAR Dataset for Reference Range Prediction and Lab Result Classification

 Overview

This dataset, LabQAR (Laboratory Question Answering with Reference Ranges), is designed to evaluate the performance of large language models (LLMs) in two crucial clinical reasoning tasks:

1. Reference Range Prediction – Given a lab test, predict the correct SI reference range.
2. Lab Result Classification – Classify a given numeric value as High, Normal, or Low based on contextual factors like specimen type, unit, gender, and age.

The dataset is structured in JSON format and consists of two main sets, each aligned with different types of question-answering tasks to assess LLMs' capabilities in clinical decision support.

---

Dataset Structure

 📁 Set 1: Reference Range Prediction

Task: Predict the correct lower and upper bound SI reference range values.

```json
[
  {
    "ID": 1,
    "Question": "For the lab test 'Acetaminophen' measuring in 'μmol/L' in Specimen 'Serum, plasma' for 'any gender' and 'any age group', what is the correct lower and upper bound range values in SI reference range?",
    "Answer": "70–200"
  }
]
```

 📁 Set 2: Lab Result Classification

Task: Classify a numeric lab test result as `High`, `Normal`, or `Low`.

```json
[
  {
    "ID": 1,
    "Question": "For the lab test 'Acetaminophen' measuring in 'μmol/L' in Specimen 'Serum, plasma' for 'any gender' and 'any age group', a value in 'SI reference range' is 341.62. Is the lab test result?",
    "Choices": "\nA: High\nB: Normal\nC: Low",
    "Answer": "A"
  }
]
```

---

Model Evaluation

We assessed five prominent LLMs:

- LLaMA 3.1 (locally deployed via Hugging Face and LangChain)
- GatorTronGPT
- GPT-3.5
- GPT-4
- GPT-4o

Among these, GPT-4o achieved the highest accuracy in both exact range prediction and classification accuracy, including within tolerance margins of ±10% and ±20% of the reference range.

---

Environment and Deployment

Local LLM: LLaMA 3.1

- Source: Hugging Face
- Framework: LangChain (`langchain==0.3.25`)
- Hardware: NVIDIA RTX A6000 (48GB VRAM)
- Inference Settings: `temperature=0` for deterministic outputs

Remote LLMs via API

- OpenAI Models (GPT-3.5, GPT-4, GPT-4o) accessed via [OpenAI API](https://platform.openai.com/docs)
- Ensure valid API key and environment variables are configured

---

 📂 Files Included

| Filename                  | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `set1_reference_range.json` | JSON data for reference range prediction questions                        |
| `set2_classification.json`  | JSON data for lab value classification questions                         |
| `annotation_guidelines.pdf`| Detailed instructions followed by annotators during data curation       |
| `requirements.txt`         | Python package versions for environment replication and model inference |

---

 🔍 Suggested Evaluation Metrics

- Exact Match Accuracy (for Set 1)
- Classification Accuracy (for Set 2)
- Tolerance-Based Match: Acceptable predictions within ±10% and ±20% of true range (for Set 1)
- F1 Score / Confusion Matrix (for Set 2)

---

 🧠 Example Use Cases

- Evaluating LLM reasoning in real-world clinical decision-making
- Fine-tuning models for lab result comprehension
- Building AI-assisted diagnostic agents using LangChain or RAG
