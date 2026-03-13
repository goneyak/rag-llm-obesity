content = """# RAG + LLM Obesity Management Recommendation System

This project analyzes obesity risk using behavioral and lifestyle features, then combines retrieval-augmented generation (RAG) with Gemini to produce personalized obesity management recommendations.

## Project Structure

```
rag-llm-obesity/
├── code/
│   ├── 1_preprocessing.ipynb   # EDA + encoding/scaling + artifact export
│   ├── 2_clustering.ipynb      # KMedoids clustering + silhouette analysis + cluster summaries
│   ├── 3_prediction.ipynb      # Binary obesity prediction using behavioral features only
│   └── 4_rag_llm.ipynb         # Guideline RAG + Gemini recommendations + BERTScore
├── data/
│   ├── obesity.csv             # Source dataset (2,111 samples, 17 features)
│   └── guidelines/
│       └── obesity_management_guidelines.txt
├── artifacts/                  # Auto-generated at runtime (gitignored)
├── .env.example
└── requirements.txt
```

## Design Principle: Prevent Data Leakage

The main issue in the original codebase was that `Height` and `Weight` are direct upstream variables of the target `NObeyesdad`.

```
BMI = Weight / Height^2  ->  NObeyesdad (obesity class)
```

If those variables are used as features, the model effectively relearns the BMI formula and can report 99%+ accuracy without having real predictive value. The rewritten pipeline uses only 14 behavioral and lifestyle features.

| Feature | Description |
|------|------|
| Age | Age |
| Gender | Gender |
| CALC | Alcohol consumption frequency |
| FAVC | Frequent high-calorie food consumption |
| FCVC | Vegetable consumption frequency (1-3) |
| NCP | Number of main meals per day |
| SCC | Calorie monitoring |
| SMOKE | Smoking status |
| CH2O | Daily water intake (L) |
| family_history_with_overweight | Family history of overweight |
| FAF | Physical activity frequency (0-3) |
| TUE | Daily technology use (hr/day) |
| CAEC | Snack consumption frequency between meals |
| MTRANS | Main mode of transportation |

## How To Run

### 1. Environment Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Set GEMINI_API_KEY in .env
```

### 2. Run the notebooks in order

The notebooks must be run in sequence.

```
1_preprocessing   -> artifacts/encoders.pkl, scaler.pkl, processed_data.csv
2_clustering      -> artifacts/cluster_model.pkl, cluster_summaries.json
3_prediction      -> artifacts/prediction_model.pkl
4_rag_llm         -> cluster-specific guideline recommendations (Gemini API required)
```

## Tech Stack

- **Clustering**: KMedoids (scikit-learn-extra) with silhouette-based model selection
- **Prediction**: Logistic Regression using behavioral features only with `class_weight=balanced`
- **Embeddings**: SentenceTransformer `all-MiniLM-L6-v2`
- **LLM**: Google Gemini 1.5 Flash
- **RAG retrieval**: Cosine similarity over normalized embeddings
- **Evaluation**: BERTScore (`distilbert-base-uncased`)

## Dataset

- **Source**: [UCI Obesity Dataset](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)
- 2,111 samples / 17 features / 7-class obesity target
- Note: roughly 77% of the dataset is reported to be synthetically generated using SMOTE
"""

with open("/Users/cocoxoxo/rag-llm-obesity/README.md", "w", encoding="utf-8") as file:
    file.write(content)

print("README.md written")
