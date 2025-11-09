# RAG-LLM System for Obesity Risk Stratification and Personalized Treatment Reasoning

This repository presents a system that integrates unsupervised phenotypic clustering, risk prediction modeling, and Retrieval-Augmented Generation (RAG) to support personalized obesity treatment reasoning. The system leverages structured population-level patterns alongside clinical guideline text to generate context-specific recommendations.

## 1. Research Objective

Obesity is a heterogeneous condition influenced by behavioral, metabolic, and socio-environmental factors. Conventional BMI- or weight-based classifications lack the resolution required for individualized clinical decision-making. The objective of this project is to:

1. Identify clinically meaningful subgroups of individuals exhibiting distinct obesity-related phenotypes.
2. Construct a risk prediction model capable of estimating an individual's likelihood of developing obesity.
3. Integrate subgroup-level behavioral characteristics with clinical guideline evidence via RAG to support interpretable and tailored recommendations.

## 2. Dataset

The dataset contains lifestyle, dietary, physiological, and demographic features associated with obesity progression.

- Features include dietary frequency, physical activity level, sedentary time, smoking status, hydration level, meal timing, and familial obesity history.
- Target variable: Multi-class obesity level (e.g., Normal Weight, Overweight, Obesity Type I–III).
- All features were cleaned, encoded, and standardized prior to analysis.

## 3. Methodology

### 3.1 Preprocessing

- Encoding categorical lifestyle and dietary variables
- Standardization of continuous features
- Removal of redundant or low-variance features

### 3.2 Clustering

- K-Means clustering (k = 10) was used to derive lifestyle- and behavior-based phenotypes.
- Cluster profiles were interpreted by examining average feature distributions.
- Examples of derived profiles include:
  - High sedentary behavior with caloric excess
  - Active individuals with high caloric intake
  - Irregular meal schedule and sleep instability
  - Familial predisposition dominant groups

### 3.3 Risk Prediction

- A multinomial logistic regression model was trained to classify obesity levels.
- Model performance was evaluated using accuracy and confusion matrices.
- The model generates probabilistic risk profiles rather than binary decisions.

### 3.4 Retrieval-Augmented Generation (RAG)

- Obesity treatment guidelines were embedded and indexed.
- For a given individual:
  1. Cluster membership is determined.
  2. Relevant guideline sections are retrieved.
  3. Cluster characteristics and guideline evidence are synthesized using a controlled LLM prompting method.

- This approach ensures traceability to both group-level behavioral patterns and authoritative clinical sources.

## 4. System Workflow

- Input Patient Features
- Data Standardization
- Cluster Assignment (K-Means)
- Risk Prediction (Logistic Regression)
- RAG Query:
  - Retrieve relevant guideline text
  - Retrieve cluster-specific behavioral profile
- LLM-Based Synthesis (Controlled Prompt Template)
- Personalized Recommendation Output

## 5. Results

- Clustering identified distinct behavioral phenotypes not captured by BMI alone.
- The logistic regression model achieved strong multi-class prediction performance, producing interpretable risk distributions.
- The RAG pipeline generated recommendations that aligned with clinical guidelines and reflected behavioral subgroup context.

## 6. Repository Structure
```
rag-llm-obesity/
├── data/
│   └── obesity.csv
├── code/
│   ├── preprocessing.ipynb
│   ├── clustering.ipynb
│   ├── prediction.ipynb
│   └── rag_llm.ipynb
├── doc/
│   └── project_report.pdf
├── requirements.txt
└── README.md
```

## 7. Installation
pip install -r requirements.txt


## 8. Reproducibility

Run the analysis in the following sequence:

1. `code/preprocessing.ipynb`
2. `code/clustering.ipynb`
3. `code/prediction.ipynb`
3. `code/rag_llm.ipynb`

## 9. Future Work

- External validation on clinical patient cohorts
- Integration of laboratory and longitudinal clinical measurements
- Evaluation of treatment recommendation effectiveness through counterfactual analysis








mini or OpenAI) for `4_rag_llm.ipynb`
