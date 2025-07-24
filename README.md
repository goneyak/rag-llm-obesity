# 🧠 Obesity Risk Prediction and RAG-based Personalized Recommendation

This project combines clustering and predictive modeling with a RAG (Retrieval-Augmented Generation) system to analyze and provide insights for obesity treatment strategies.

---

## 📂 Project Structure

```
obesity-rag-project/
├── Data/
│   └── obesity.csv                 # Original dataset (cleaned)
├── Code/
│   ├── 1_preprocessing.ipynb      # Data cleaning and preprocessing
│   ├── 2_clustering.ipynb         # KMeans clustering (k=10) and cluster profiling
│   ├── 3_prediction.ipynb         # Obesity level prediction using Logistic Regression
│   └── 4_rag_llm.ipynb            # RAG+LLM: Cluster-based health advice generation
├── requirements.txt               # Python dependencies
└── README.md
```

---

## 🚀 Key Features

- **📊 Data Preprocessing**: Encoding, scaling, handling categorical features  
- **🔍 Clustering**: KMeans-based segmentation of users for personalized insights  
- **🧠 Prediction**: Logistic Regression model to estimate obesity risk  
- **🧾 RAG with LLM**: PDF-based medical guideline + cluster context → treatment advice

---

## 🛠️ Tech Stack

- `Python`, `scikit-learn`, `pandas`, `matplotlib`
- `Sentence-BERT`, `BERTScore`, `PyPDF2`
- (Optional) `OpenAI` / `Gemini API` for medical LLMs

---

## ✅ Usage

1. Clone the repository:

```bash
git clone https://github.com/goneyak/obesity-rag-project.git
cd obesity-rag-project
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run each notebook in order:

- `Code/1_preprocessing.ipynb`  
- `Code/2_clustering.ipynb`  
- `Code/3_prediction.ipynb`  
- `Code/4_rag_llm.ipynb`

---

## 📌 Notes

- Data is cleaned and ready to use (`obesity.csv`)
- You may need your own LLM API key (e.g., Gemini or OpenAI) for `4_rag_llm.ipynb`
