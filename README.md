# RAG-LLM System for Obesity Risk Stratification and Personalized Treatment Reasoning

Research portfolio project combining clustering, risk prediction, and guideline-grounded RAG to support interpretable obesity treatment reasoning.

## 1. Research Objective

Obesity is heterogeneous across behavioral, metabolic, and socio-environmental dimensions. This project explores a structured pipeline that goes beyond single-metric categorization by:

1. Identifying phenotype-like lifestyle subgroups.
2. Estimating obesity risk through supervised modeling.
3. Grounding recommendation text in retrieved guideline evidence plus subgroup context.

## 2. Dataset

The dataset contains lifestyle, dietary, physiological, and demographic features related to obesity progression.

- Data file: `data/obesity.csv`
- Target: multi-class obesity level
- Typical features: eating habits, physical activity, sedentary behavior, hydration, smoking, and family obesity history

## 3. Pipeline Summary

### 3.1 Preprocessing

- Encode categorical variables
- Standardize continuous features
- Prepare cleaned feature matrix for downstream analysis

### 3.2 Clustering

- Derive subgroup structure using unsupervised methods
- Interpret subgroup profiles from feature-level patterns

### 3.3 Risk Prediction

- Train a multinomial logistic regression classifier for obesity level estimation
- Evaluate with standard classification metrics

### 3.4 Guideline-Grounded RAG

- Embed and retrieve obesity guideline snippets
- Combine retrieved evidence with subgroup context
- Generate recommendation-style reasoning with evidence grounding

## 4. Stage Outputs

1. `code/1_preprocessing.ipynb`
- cleaned/encoded data and analysis-ready inputs

2. `code/2_clustering.ipynb`
- subgroup assignments and profile interpretation artifacts

3. `code/3_prediction.ipynb`
- obesity risk model outputs and evaluation summaries

4. `code/4_rag_llm.ipynb`
- guideline retrieval and LLM-based synthesized recommendations

## 5. Repository Structure

```text
rag-llm-obesity/
├── code/
│   ├── 1_preprocessing.ipynb
│   ├── 2_clustering.ipynb
│   ├── 3_prediction.ipynb
│   └── 4_rag_llm.ipynb
├── data/
│   └── obesity.csv
├── doc/
│   └── project_report.pdf
├── requirements.txt
└── README.md
```

## 6. Installation

Run from repository root:

```bash
pip install -r requirements.txt
```

## 7. Reproducibility

Open notebooks from repository root and run in this exact order:

1. `code/1_preprocessing.ipynb`
2. `code/2_clustering.ipynb`
3. `code/3_prediction.ipynb`
4. `code/4_rag_llm.ipynb`

## 8. Environment Notes

- `code/4_rag_llm.ipynb` expects an LLM API key via environment variable:
- `GEMINI_API_KEY`
- If you use a different provider, update only the notebook runtime configuration (do not hardcode secrets in the notebook).

## 9. Prototype Notes and Limitations

- This is a research and educational portfolio prototype.
- The workflow is designed for interpretability and evidence grounding, not direct clinical deployment.
- Recommendations are generated from retrieved guideline text plus subgroup context, but this repository does not provide external clinical validation.

## 10. Future Work

- External validation on independent cohorts
- Integration of longitudinal and laboratory features
- More rigorous evaluation of recommendation quality and robustness
