"""Generate clean English-language notebooks for the rag-llm-obesity project."""

from __future__ import annotations

import json
import os
import uuid


def cell_id() -> str:
    return str(uuid.uuid4())[:8]


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id(),
        "metadata": {},
        "source": source,
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": cell_id(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source,
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "cells": cells,
    }


def write_notebook(path: str, cells: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(notebook(cells), file, ensure_ascii=False, indent=1)
    print(f"Created: {path}")


preprocessing_cells = [
    markdown(
        "# 1. Obesity Data Preprocessing\n\n"
        "**Goal:** Load the raw dataset, perform exploratory analysis, and save reusable preprocessing artifacts for downstream notebooks.\n\n"
        "**Core design rule: prevent data leakage**\n"
        "- `Height` and `Weight` are direct upstream variables of the target via BMI.\n"
        "- Using them as features lets the model relearn the obesity label definition.\n"
        "- This rewritten pipeline uses **behavioral and lifestyle features only**."
    ),
    code(
        "import os\n"
        "import pickle\n"
        "\n"
        "import matplotlib.pyplot as plt\n"
        "import pandas as pd\n"
        "import seaborn as sns\n"
        "\n"
        "DATA_PATH = '../data/obesity.csv'\n"
        "ARTIFACTS_DIR = '../artifacts'\n"
        "os.makedirs(ARTIFACTS_DIR, exist_ok=True)"
    ),
    code(
        "data = pd.read_csv(DATA_PATH)\n"
        "print(f'Shape: {data.shape}')\n"
        "print(f'Columns: {list(data.columns)}')\n"
        "data.head()"
    ),
    markdown("## Target distribution"),
    code(
        "obesity_order = [\n"
        "    'Insufficient_Weight', 'Normal_Weight',\n"
        "    'Overweight_Level_I', 'Overweight_Level_II',\n"
        "    'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'\n"
        "]\n"
        "counts = data['NObeyesdad'].value_counts().reindex(obesity_order)\n"
        "plt.figure(figsize=(10, 4))\n"
        "sns.barplot(x=obesity_order, y=counts.values, palette='RdYlGn_r')\n"
        "plt.xticks(rotation=30, ha='right', fontsize=9)\n"
        "plt.title('Obesity Label Distribution')\n"
        "plt.ylabel('Count')\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "print(counts)"
    ),
    markdown(
        "## Feature selection\n\n"
        "This project keeps only behavioral and lifestyle predictors and excludes `Height` and `Weight`."
    ),
    code(
        "CATEGORICAL_FEATURES = [\n"
        "    'Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE',\n"
        "    'family_history_with_overweight', 'CAEC', 'MTRANS'\n"
        "]\n"
        "CONTINUOUS_FEATURES = ['Age', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']\n"
        "TARGET = 'NObeyesdad'\n"
        "\n"
        "print('Selected feature groups:')\n"
        "print(f'  Categorical: {CATEGORICAL_FEATURES}')\n"
        "print(f'  Continuous: {CONTINUOUS_FEATURES}')\n"
        "print('  Excluded due to leakage: [Height, Weight]')"
    ),
    markdown("## EDA: continuous feature distributions"),
    code(
        "fig, axes = plt.subplots(2, 3, figsize=(14, 7))\n"
        "for axis, column in zip(axes.flatten(), CONTINUOUS_FEATURES):\n"
        "    sns.histplot(data[column], kde=True, ax=axis, bins=25, color='steelblue')\n"
        "    axis.set_title(column)\n"
        "plt.suptitle('Behavioral Continuous Features', fontsize=13, y=1.02)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    markdown("## EDA: categorical feature distributions"),
    code(
        "fig, axes = plt.subplots(2, 4, figsize=(18, 8))\n"
        "for axis, column in zip(axes.flatten(), CATEGORICAL_FEATURES):\n"
        "    value_counts = data[column].value_counts()\n"
        "    sns.barplot(x=value_counts.index, y=value_counts.values, ax=axis, palette='Set2')\n"
        "    axis.set_title(column)\n"
        "    axis.tick_params(axis='x', rotation=30)\n"
        "plt.suptitle('Categorical Feature Distributions', fontsize=13, y=1.01)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    markdown(
        "## Encode categorical variables\n\n"
        "- Binary features use fixed dictionary mappings\n"
        "- Ordered features use explicit ordinal mappings\n"
        "- The mappings are stored in `artifacts/encoders.pkl` for reuse during inference"
    ),
    code(
        "BINARY_MAP = {\n"
        "    'Gender': {'Female': 0, 'Male': 1},\n"
        "    'FAVC': {'no': 0, 'yes': 1},\n"
        "    'SCC': {'no': 0, 'yes': 1},\n"
        "    'SMOKE': {'no': 0, 'yes': 1},\n"
        "    'family_history_with_overweight': {'no': 0, 'yes': 1},\n"
        "}\n"
        "ORDINAL_MAP = {\n"
        "    'CALC': ['no', 'Sometimes', 'Frequently', 'Always'],\n"
        "    'CAEC': ['no', 'Sometimes', 'Frequently', 'Always'],\n"
        "    'MTRANS': ['Bike', 'Walking', 'Public_Transportation', 'Motorbike', 'Automobile'],\n"
        "}\n"
        "\n"
        "processed = data.copy()\n"
        "for column, mapping in BINARY_MAP.items():\n"
        "    processed[column] = processed[column].map(mapping)\n"
        "for column, order in ORDINAL_MAP.items():\n"
        "    processed[column] = processed[column].map({value: index for index, value in enumerate(order)})\n"
        "\n"
        "encoders = {'binary': BINARY_MAP, 'ordinal': ORDINAL_MAP}\n"
        "with open(f'{ARTIFACTS_DIR}/encoders.pkl', 'wb') as file:\n"
        "    pickle.dump(encoders, file)\n"
        "\n"
        "print('Saved artifacts/encoders.pkl')\n"
        "processed[CATEGORICAL_FEATURES].head()"
    ),
    markdown("## Scale continuous features"),
    code(
        "from sklearn.preprocessing import StandardScaler\n"
        "\n"
        "scaler = StandardScaler()\n"
        "processed[CONTINUOUS_FEATURES] = scaler.fit_transform(processed[CONTINUOUS_FEATURES])\n"
        "with open(f'{ARTIFACTS_DIR}/scaler.pkl', 'wb') as file:\n"
        "    pickle.dump(scaler, file)\n"
        "\n"
        "print('Saved artifacts/scaler.pkl')\n"
        "print('Scaled feature means:')\n"
        "print(processed[CONTINUOUS_FEATURES].mean().round(4).to_dict())"
    ),
    markdown("## Correlation analysis"),
    code(
        "feature_frame = processed[CATEGORICAL_FEATURES + CONTINUOUS_FEATURES]\n"
        "correlation_matrix = feature_frame.corr()\n"
        "\n"
        "plt.figure(figsize=(10, 8))\n"
        "sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.4)\n"
        "plt.title('Behavioral Feature Correlation Heatmap')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    markdown("## Save the processed dataset"),
    code(
        "save_columns = CATEGORICAL_FEATURES + CONTINUOUS_FEATURES + [TARGET]\n"
        "processed[save_columns].to_csv(f'{ARTIFACTS_DIR}/processed_data.csv', index=False)\n"
        "print(f'Saved artifacts/processed_data.csv with shape {processed[save_columns].shape}')\n"
        "processed[save_columns].head()"
    ),
]


clustering_cells = [
    markdown(
        "# 2. Obesity Pattern Clustering\n\n"
        "**Goal:** Group similar behavioral profiles using KMedoids and summarize each cluster in plain language.\n\n"
        "**Input:** `artifacts/processed_data.csv`\n"
        "**Outputs:** `artifacts/cluster_model.pkl`, `artifacts/cluster_summaries.json`, `artifacts/obesity_with_clusters.csv`"
    ),
    code(
        "import json\n"
        "import os\n"
        "import pickle\n"
        "\n"
        "import matplotlib.pyplot as plt\n"
        "import pandas as pd\n"
        "import seaborn as sns\n"
        "from sklearn.decomposition import PCA\n"
        "from sklearn.metrics import silhouette_score\n"
        "from sklearn_extra.cluster import KMedoids\n"
        "\n"
        "os.environ['OPENBLAS_NUM_THREADS'] = '1'\n"
        "ARTIFACTS_DIR = '../artifacts'"
    ),
    code(
        "CATEGORICAL_FEATURES = [\n"
        "    'Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE',\n"
        "    'family_history_with_overweight', 'CAEC', 'MTRANS'\n"
        "]\n"
        "CONTINUOUS_FEATURES = ['Age', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']\n"
        "FEATURES = CATEGORICAL_FEATURES + CONTINUOUS_FEATURES\n"
        "cluster_frame = pd.read_csv(f'{ARTIFACTS_DIR}/processed_data.csv')\n"
        "X = cluster_frame[FEATURES].values\n"
        "print(f'Loaded {len(cluster_frame)} rows and {len(FEATURES)} features')\n"
        "cluster_frame.head()"
    ),
    markdown("## Select the number of clusters with silhouette analysis"),
    code(
        "scores = []\n"
        "for k in range(4, 11):\n"
        "    model = KMedoids(n_clusters=k, random_state=42, max_iter=300)\n"
        "    labels = model.fit_predict(X)\n"
        "    score = silhouette_score(X, labels, sample_size=1000, random_state=42)\n"
        "    scores.append((k, score))\n"
        "    print(f'k={k}, silhouette={score:.4f}')\n"
        "\n"
        "best_k, best_score = max(scores, key=lambda item: item[1])\n"
        "print(f'Best k: {best_k}')\n"
        "\n"
        "plt.figure(figsize=(8, 4))\n"
        "plt.plot([item[0] for item in scores], [item[1] for item in scores], 'o-', color='steelblue')\n"
        "plt.axvline(best_k, color='red', linestyle='--', label=f'Best k={best_k}')\n"
        "plt.xlabel('Number of clusters')\n"
        "plt.ylabel('Silhouette score')\n"
        "plt.title('KMedoids Silhouette Analysis')\n"
        "plt.legend()\n"
        "plt.grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    markdown("## Fit the final clustering model"),
    code(
        "final_model = KMedoids(n_clusters=best_k, random_state=42, max_iter=500)\n"
        "cluster_frame['cluster'] = final_model.fit_predict(X)\n"
        "\n"
        "with open(f'{ARTIFACTS_DIR}/cluster_model.pkl', 'wb') as file:\n"
        "    pickle.dump({'model': final_model, 'n_clusters': best_k, 'features': FEATURES}, file)\n"
        "\n"
        "print('Saved artifacts/cluster_model.pkl')\n"
        "print(cluster_frame['cluster'].value_counts().sort_index())"
    ),
    markdown("## PCA projection"),
    code(
        "pca = PCA(n_components=2)\n"
        "projection = pca.fit_transform(X)\n"
        "projection_frame = pd.DataFrame(projection, columns=['PC1', 'PC2'])\n"
        "projection_frame['cluster'] = cluster_frame['cluster'].values\n"
        "projection_frame['NObeyesdad'] = cluster_frame['NObeyesdad'].values\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n"
        "sns.scatterplot(data=projection_frame, x='PC1', y='PC2', hue='cluster', palette='tab10', s=30, alpha=0.7, ax=axes[0])\n"
        "axes[0].set_title('Clusters in PCA Space')\n"
        "\n"
        "obesity_order = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']\n"
        "sns.scatterplot(data=projection_frame, x='PC1', y='PC2', hue='NObeyesdad', hue_order=obesity_order, palette='RdYlGn_r', s=30, alpha=0.7, ax=axes[1])\n"
        "axes[1].set_title('True Obesity Labels in PCA Space')\n"
        "axes[1].legend(fontsize=7, loc='upper right')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    markdown("## Cluster-level summary statistics"),
    code(
        "raw = pd.read_csv('../data/obesity.csv')\n"
        "raw['cluster'] = cluster_frame['cluster'].values\n"
        "raw['BMI'] = raw['Weight'] / (raw['Height'] ** 2)\n"
        "display_columns = ['Age', 'BMI', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']\n"
        "cluster_stats = raw.groupby('cluster')[display_columns].mean().round(2)\n"
        "cluster_stats['dominant_label'] = raw.groupby('cluster')['NObeyesdad'].agg(lambda values: values.value_counts().index[0])\n"
        "cluster_stats['count'] = raw.groupby('cluster').size()\n"
        "cluster_stats"
    ),
    markdown("## Generate plain-English cluster summaries"),
    code(
        "summaries = []\n"
        "for cluster_id in sorted(raw['cluster'].unique()):\n"
        "    subset = raw[raw['cluster'] == cluster_id]\n"
        "    means = subset[display_columns].mean().round(2)\n"
        "    dominant_label = subset['NObeyesdad'].value_counts().index[0]\n"
        "    family_history_rate = (subset['family_history_with_overweight'] == 'yes').mean() * 100\n"
        "    summary_text = (\n"
        "        f\"Cluster {cluster_id} (n={len(subset)}): average age {means['Age']:.0f} years, \"\n"
        "        f\"average BMI {means['BMI']:.1f}, vegetable frequency {means['FCVC']:.2f}/3, \"\n"
        "        f\"meals per day {means['NCP']:.2f}, water intake {means['CH2O']:.2f} L/day, \"\n"
        "        f\"physical activity {means['FAF']:.2f}/3, technology use {means['TUE']:.2f} hr/day, \"\n"
        "        f\"family history of overweight {family_history_rate:.0f}%. \"\n"
        "        f\"Most common obesity label: {dominant_label}.\"\n"
        "    )\n"
        "    summaries.append({\n"
        "        'cluster': int(cluster_id),\n"
        "        'n': int(len(subset)),\n"
        "        'dominant_label': dominant_label,\n"
        "        'summary': summary_text,\n"
        "    })\n"
        "    print(summary_text)\n"
        "\n"
        "with open(f'{ARTIFACTS_DIR}/cluster_summaries.json', 'w', encoding='utf-8') as file:\n"
        "    json.dump(summaries, file, indent=2, ensure_ascii=False)\n"
        "print('Saved artifacts/cluster_summaries.json')"
    ),
    code(
        "cluster_frame.to_csv(f'{ARTIFACTS_DIR}/obesity_with_clusters.csv', index=False)\n"
        "print('Saved artifacts/obesity_with_clusters.csv')"
    ),
]


prediction_cells = [
    markdown(
        "# 3. Obesity Risk Prediction\n\n"
        "**Goal:** Predict binary obesity risk using behavioral and lifestyle features only.\n\n"
        "`Height`, `Weight`, and `BMI` are intentionally excluded to avoid target leakage."
    ),
    code(
        "import pickle\n"
        "\n"
        "import matplotlib.pyplot as plt\n"
        "import pandas as pd\n"
        "import seaborn as sns\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve\n"
        "from sklearn.model_selection import cross_val_score, train_test_split\n"
        "\n"
        "ARTIFACTS_DIR = '../artifacts'"
    ),
    code(
        "prediction_frame = pd.read_csv(f'{ARTIFACTS_DIR}/processed_data.csv')\n"
        "CATEGORICAL_FEATURES = ['Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight', 'CAEC', 'MTRANS']\n"
        "CONTINUOUS_FEATURES = ['Age', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']\n"
        "FEATURES = CATEGORICAL_FEATURES + CONTINUOUS_FEATURES\n"
        "obese_labels = ['Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']\n"
        "prediction_frame['is_obese'] = prediction_frame['NObeyesdad'].isin(obese_labels).astype(int)\n"
        "print(prediction_frame['is_obese'].value_counts())\n"
        "print(f\"Obesity rate: {prediction_frame['is_obese'].mean():.2%}\")"
    ),
    markdown("## Train/test split"),
    code(
        "X = prediction_frame[FEATURES]\n"
        "y = prediction_frame['is_obese']\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n"
        "print(f'Train rows: {len(X_train)} | Test rows: {len(X_test)}')"
    ),
    markdown("## Train logistic regression"),
    code(
        "classifier = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)\n"
        "classifier.fit(X_train, y_train)\n"
        "cv_auc_scores = cross_val_score(classifier, X_train, y_train, cv=5, scoring='roc_auc')\n"
        "print(f'5-fold CV ROC AUC: {cv_auc_scores.mean():.4f} +/- {cv_auc_scores.std():.4f}')"
    ),
    markdown("## Evaluate the classifier"),
    code(
        "predicted_labels = classifier.predict(X_test)\n"
        "predicted_probabilities = classifier.predict_proba(X_test)[:, 1]\n"
        "accuracy = accuracy_score(y_test, predicted_labels)\n"
        "roc_auc = roc_auc_score(y_test, predicted_probabilities)\n"
        "print(f'Accuracy: {accuracy:.4f}')\n"
        "print(f'ROC AUC: {roc_auc:.4f}')\n"
        "print()\n"
        "print(classification_report(y_test, predicted_labels, target_names=['Not Obese', 'Obese']))"
    ),
    code(
        "figure, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
        "matrix = confusion_matrix(y_test, predicted_labels)\n"
        "ConfusionMatrixDisplay(matrix, display_labels=['Not Obese', 'Obese']).plot(ax=axes[0], cmap='Blues')\n"
        "axes[0].set_title('Confusion Matrix')\n"
        "false_positive_rate, true_positive_rate, _ = roc_curve(y_test, predicted_probabilities)\n"
        "axes[1].plot(false_positive_rate, true_positive_rate, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')\n"
        "axes[1].plot([0, 1], [0, 1], color='navy', linestyle='--')\n"
        "axes[1].set_xlabel('False Positive Rate')\n"
        "axes[1].set_ylabel('True Positive Rate')\n"
        "axes[1].set_title('ROC Curve')\n"
        "axes[1].legend()\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    markdown("## Feature importance via model coefficients"),
    code(
        "coefficient_frame = pd.DataFrame({'Feature': FEATURES, 'Coefficient': classifier.coef_[0]})\n"
        "coefficient_frame = coefficient_frame.sort_values('Coefficient', key=abs, ascending=False)\n"
        "plt.figure(figsize=(10, 5))\n"
        "colors = ['#e74c3c' if value > 0 else '#3498db' for value in coefficient_frame['Coefficient']]\n"
        "sns.barplot(x='Coefficient', y='Feature', data=coefficient_frame, palette=colors)\n"
        "plt.axvline(0, color='black', linewidth=0.8)\n"
        "plt.title('Logistic Regression Coefficients')\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "coefficient_frame"
    ),
    markdown("## Save the trained model"),
    code(
        "with open(f'{ARTIFACTS_DIR}/prediction_model.pkl', 'wb') as file:\n"
        "    pickle.dump({'model': classifier, 'features': FEATURES, 'obese_labels': obese_labels}, file)\n"
        "print('Saved artifacts/prediction_model.pkl')"
    ),
    markdown("## Inference helper"),
    code(
        "def predict_obesity_risk(user_input: dict) -> dict:\n"
        "    with open(f'{ARTIFACTS_DIR}/encoders.pkl', 'rb') as file:\n"
        "        encoders = pickle.load(file)\n"
        "    with open(f'{ARTIFACTS_DIR}/scaler.pkl', 'rb') as file:\n"
        "        scaler = pickle.load(file)\n"
        "    with open(f'{ARTIFACTS_DIR}/prediction_model.pkl', 'rb') as file:\n"
        "        saved_model = pickle.load(file)\n"
        "\n"
        "    row = dict(user_input)\n"
        "    for column, mapping in encoders['binary'].items():\n"
        "        if column in row:\n"
        "            row[column] = mapping[row[column]]\n"
        "    for column, order in encoders['ordinal'].items():\n"
        "        if column in row:\n"
        "            row[column] = {value: index for index, value in enumerate(order)}[row[column]]\n"
        "\n"
        "    categorical = ['Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight', 'CAEC', 'MTRANS']\n"
        "    continuous = ['Age', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']\n"
        "    scaled_continuous = scaler.transform([[row[column] for column in continuous]])[0]\n"
        "    transformed = {column: scaled_continuous[index] for index, column in enumerate(continuous)}\n"
        "    transformed.update({column: row[column] for column in categorical})\n"
        "\n"
        "    feature_frame = pd.DataFrame([transformed])[saved_model['features']]\n"
        "    probability = saved_model['model'].predict_proba(feature_frame)[0][1]\n"
        "    if probability < 0.30:\n"
        "        risk_label = 'Low risk'\n"
        "    elif probability < 0.60:\n"
        "        risk_label = 'Moderate risk'\n"
        "    else:\n"
        "        risk_label = 'High risk'\n"
        "    return {'risk_score': round(float(probability), 4), 'label': risk_label}\n"
        "\n"
        "example_input = {\n"
        "    'Age': 28, 'Gender': 'Male', 'CALC': 'Sometimes', 'FAVC': 'yes',\n"
        "    'FCVC': 1, 'NCP': 3, 'SCC': 'yes', 'SMOKE': 'no', 'CH2O': 1,\n"
        "    'family_history_with_overweight': 'yes', 'FAF': 0, 'TUE': 2,\n"
        "    'CAEC': 'Frequently', 'MTRANS': 'Automobile'\n"
        "}\n"
        "print(predict_obesity_risk(example_input))"
    ),
]


rag_cells = [
    markdown(
        "# 4. RAG + LLM Obesity Guideline Recommendation\n\n"
        "Pipeline overview:\n"
        "1. Load guideline files from `data/guidelines/`\n"
        "2. Split them into sentence-based chunks\n"
        "3. Embed all chunks with SentenceTransformer\n"
        "4. Use cluster summaries as retrieval queries\n"
        "5. Send retrieved evidence and the profile summary to Gemini\n"
        "6. Evaluate generated recommendations with BERTScore"
    ),
    code("%pip install -q sentence-transformers PyPDF2 nltk bert-score google-generativeai python-dotenv"),
    code(
        "import glob\n"
        "import json\n"
        "import os\n"
        "\n"
        "import google.generativeai as genai\n"
        "import nltk\n"
        "import numpy as np\n"
        "import PyPDF2\n"
        "from dotenv import load_dotenv\n"
        "from nltk.tokenize import sent_tokenize\n"
        "from sentence_transformers import SentenceTransformer\n"
        "\n"
        "nltk.download('punkt', quiet=True)\n"
        "nltk.download('punkt_tab', quiet=True)\n"
        "load_dotenv('../.env')\n"
        "GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')\n"
        "if not GEMINI_API_KEY:\n"
        "    raise ValueError('GEMINI_API_KEY not found. Create .env from .env.example and set your key.')\n"
        "genai.configure(api_key=GEMINI_API_KEY)\n"
        "ARTIFACTS_DIR = '../artifacts'\n"
        "GUIDELINES_DIR = '../data/guidelines'"
    ),
    markdown("## Load guideline files and build text chunks"),
    code(
        "def load_text_from_file(path: str) -> str:\n"
        "    if path.endswith('.pdf'):\n"
        "        pages = []\n"
        "        with open(path, 'rb') as file:\n"
        "            reader = PyPDF2.PdfReader(file)\n"
        "            for page in reader.pages:\n"
        "                text = page.extract_text()\n"
        "                if text:\n"
        "                    pages.append(text)\n"
        "        return ' '.join(pages)\n"
        "    with open(path, 'r', encoding='utf-8') as file:\n"
        "        return file.read()\n"
        "\n"
        "def make_chunks(text: str, window: int = 3, stride: int = 2) -> list[str]:\n"
        "    sentences = [sentence.strip() for sentence in sent_tokenize(text) if len(sentence.strip()) > 20]\n"
        "    chunks = []\n"
        "    for index in range(0, len(sentences), stride):\n"
        "        chunk = ' '.join(sentences[index:index + window])\n"
        "        if chunk:\n"
        "            chunks.append(chunk)\n"
        "    return chunks\n"
        "\n"
        "all_chunks = []\n"
        "all_sources = []\n"
        "files = glob.glob(f'{GUIDELINES_DIR}/*.txt') + glob.glob(f'{GUIDELINES_DIR}/*.pdf')\n"
        "if not files:\n"
        "    raise FileNotFoundError(f'No guideline files found in {GUIDELINES_DIR}')\n"
        "for path in files:\n"
        "    source_name = os.path.basename(path)\n"
        "    text = load_text_from_file(path)\n"
        "    chunks = make_chunks(text)\n"
        "    all_chunks.extend(chunks)\n"
        "    all_sources.extend([source_name] * len(chunks))\n"
        "    print(f'{source_name}: {len(chunks)} chunks')\n"
        "print(f'Total chunks: {len(all_chunks)}')"
    ),
    markdown("## Generate and save chunk embeddings"),
    code(
        "embedding_model = SentenceTransformer('all-MiniLM-L6-v2')\n"
        "chunk_embeddings = embedding_model.encode(all_chunks, batch_size=64, show_progress_bar=True, normalize_embeddings=True)\n"
        "np.savez(f'{ARTIFACTS_DIR}/rag_chunks.npz', embeddings=chunk_embeddings, chunks=np.array(all_chunks), sources=np.array(all_sources))\n"
        "print(f'Saved artifacts/rag_chunks.npz with shape {chunk_embeddings.shape}')"
    ),
    markdown("## Load cluster summaries"),
    code(
        "with open(f'{ARTIFACTS_DIR}/cluster_summaries.json', encoding='utf-8') as file:\n"
        "    cluster_summaries = json.load(file)\n"
        "print(f'Loaded {len(cluster_summaries)} cluster summaries')\n"
        "cluster_summaries[:2]"
    ),
    markdown("## Retrieval helper"),
    code(
        "def retrieve_chunks(query: str, top_k: int = 5) -> list[dict]:\n"
        "    stored = np.load(f'{ARTIFACTS_DIR}/rag_chunks.npz', allow_pickle=True)\n"
        "    embeddings = stored['embeddings']\n"
        "    chunks = stored['chunks'].tolist()\n"
        "    sources = stored['sources'].tolist()\n"
        "    query_embedding = embedding_model.encode([query], normalize_embeddings=True)[0]\n"
        "    scores = embeddings @ query_embedding\n"
        "    top_indices = np.argsort(scores)[::-1][:top_k]\n"
        "    return [{'chunk': chunks[index], 'source': sources[index], 'score': float(scores[index])} for index in top_indices]"
    ),
    markdown("## Gemini recommendation helper"),
    code(
        "def generate_recommendation(cluster_summary: str, top_k: int = 5) -> dict:\n"
        "    retrieved = retrieve_chunks(cluster_summary, top_k=top_k)\n"
        "    context = '\\n\\n'.join(f\"[Source: {item['source']}] {item['chunk']}\" for item in retrieved)\n"
        "    prompt = (\n"
        "        'You are an expert obesity management clinician.\\n'\n"
        "        'Use the patient group profile and guideline excerpts below to write concise, evidence-based recommendations.\\n\\n'\n"
        "        f'## Patient Group Profile\\n{cluster_summary}\\n\\n'\n"
        "        f'## Guideline Evidence\\n{context}\\n\\n'\n"
        "        'Respond with: (1) main risks, (2) 3-5 dietary actions, (3) 2-3 exercise actions, and (4) behavioral support ideas.'\n"
        "    )\n"
        "    model = genai.GenerativeModel('gemini-1.5-flash')\n"
        "    response = model.generate_content(prompt)\n"
        "    return {'summary': cluster_summary, 'retrieved': retrieved, 'recommendation': response.text}\n"
        "\n"
        "test_cluster = cluster_summaries[0]\n"
        "print(test_cluster['summary'])"
    ),
    code(
        "test_result = generate_recommendation(test_cluster['summary'], top_k=5)\n"
        "print(test_result['recommendation'])\n"
        "for item in test_result['retrieved']:\n"
        "    print(item['score'], item['source'])"
    ),
    markdown("## Generate recommendations for all clusters"),
    code(
        "import time\n"
        "\n"
        "all_results = []\n"
        "for cluster_summary in cluster_summaries:\n"
        "    print(f\"Processing cluster {cluster_summary['cluster']}\")\n"
        "    result = generate_recommendation(cluster_summary['summary'], top_k=5)\n"
        "    all_results.append({\n"
        "        'cluster': cluster_summary['cluster'],\n"
        "        'label': cluster_summary['dominant_label'],\n"
        "        'recommendation': result['recommendation'],\n"
        "    })\n"
        "    time.sleep(1)\n"
        "print(f'Generated recommendations for {len(all_results)} clusters')"
    ),
    markdown("## Evaluate recommendation quality with BERTScore"),
    code(
        "from bert_score import score as bert_score\n"
        "\n"
        "if all_results:\n"
        "    candidates = [item['recommendation'] for item in all_results]\n"
        "    references = [retrieve_chunks(cluster_summaries[index]['summary'], top_k=1)[0]['chunk'] for index in range(len(all_results))]\n"
        "    precision, recall, f1 = bert_score(candidates, references, lang='en', model_type='distilbert-base-uncased', verbose=False)\n"
        "    print(f'BERTScore precision: {precision.mean():.4f}')\n"
        "    print(f'BERTScore recall: {recall.mean():.4f}')\n"
        "    print(f'BERTScore F1: {f1.mean():.4f}')\n"
        "else:\n"
        "    print('No generated results available. Run the previous cell first.')"
    ),
]


BASE_DIR = "/Users/cocoxoxo/rag-llm-obesity/code"
write_notebook(f"{BASE_DIR}/1_preprocessing.ipynb", preprocessing_cells)
write_notebook(f"{BASE_DIR}/2_clustering.ipynb", clustering_cells)
write_notebook(f"{BASE_DIR}/3_prediction.ipynb", prediction_cells)
write_notebook(f"{BASE_DIR}/4_rag_llm.ipynb", rag_cells)

print("All notebooks generated successfully.")
