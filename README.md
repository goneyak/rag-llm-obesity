# 🧬 Obesity Precision Medicine with Clustering & RAG-LLM

This project explores a precision medicine approach to obesity by combining unsupervised clustering, obesity level prediction, and guideline-based LLM recommendations (RAG). We segment patients into clusters based on lifestyle and demographic features and deliver customized insights using medical guideline documents.

---

## 📌 Project Overview

### 🔹 1. Preprocessing
We clean and normalize the dataset, handle categorical values, remove outliers, and scale features.

**Notebook:** `[Project] 1. Obesity_preprocessing.ipynb`

### 🔹 2. Clustering for Patient Segmentation
Using KMeans and silhouette score analysis, we segment patients into 10 clusters based on risk and lifestyle profiles.

**Notebook:** `[Project] 2. Clustering_new.ipynb`

### 🔹 3. Obesity Level Prediction
A logistic regression model is trained to predict obesity level. We use accuracy and confusion matrix to evaluate performance.

**Notebook:** `[Project] 3. Obesity_prediction.ipynb`

### 🔹 4. Guideline-based RAG-LLM (Gemini)
A Retrieval-Augmented Generation (RAG) system extracts relevant content from obesity treatment guidelines (PDFs), and generates answers using the Gemini LLM. Evaluation is performed via BERTScore.

**Notebook:** `[Project] 4. pdf processing-RAG-LLM with clusters(진짜).ipynb`

---

## 🚀 How to Run

1. **Install Requirements**
```bash
pip install -r requirements.txt
```

2. **Run in Order**
   - `Obesity_preprocessing.ipynb`
   - `Clustering_new.ipynb`
   - `Obesity_prediction.ipynb`
   - `pdf processing-RAG-LLM with clusters(진짜).ipynb`

---

## 📂 Folder Structure

```
project-root/
│
├── data/                         # Input and processed datasets
├── notebooks/
│   ├── Obesity_preprocessing.ipynb
│   ├── Clustering_new.ipynb
│   ├── Obesity_prediction.ipynb
│   └── pdf processing-RAG-LLM with clusters(진짜).ipynb
├── outputs/                      # Cluster profiles, prediction results, etc.
└── README.md
```

---

## 🧠 Example Use Case

> "Given this patient’s profile (high sedentary time, low physical activity, frequent high-calorie food intake), what is their risk score and what treatment advice applies?"

1. Assigns patient to most similar cluster using KMeans
2. Returns summarized guideline advice (via Gemini) for that cluster
3. Ranks output based on content similarity (BERTScore)

---

## 🛠️ Tech Stack

- **Language**: Python
- **Libraries**: scikit-learn, pandas, matplotlib, seaborn, sentence-transformers
- **LLM**: Google Gemini Pro (via API)
- **Evaluation**: BERTScore

---

## 📄 Authors

- Goyeun Yun (OMSA, Spring 2025)