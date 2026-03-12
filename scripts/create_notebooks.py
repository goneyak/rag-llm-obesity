"""
Generate all four clean Jupyter notebooks for rag-llm-obesity project.
Run from the repo root: python scripts/create_notebooks.py
"""
import json, os, uuid

def uid():
    return str(uuid.uuid4())[:8]

def md(source):
    return {"cell_type": "markdown", "id": uid(), "metadata": {}, "source": source}

def code(source):
    return {"cell_type": "code", "id": uid(), "metadata": {},
            "execution_count": None, "outputs": [], "source": source}

def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "cells": cells,
    }

def write(path, notebook):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    print(f"  Created: {path}")

# ---------------------------------------------------------------------------
# Notebook 1 – Preprocessing
# ---------------------------------------------------------------------------
nb1_cells = [
    md("# 1. 비만 데이터 전처리 (Preprocessing)\n\n"
       "**목적:** 원본 데이터를 로드하고 EDA를 수행한 뒤, 이후 노트북에서 재사용 가능한\n"
       "인코더(`encoders.pkl`)와 스케일러(`scaler.pkl`)를 저장합니다.\n\n"
       "**핵심 설계 원칙 – 데이터 누수 방지:**\n"
       "- `Height`, `Weight`는 타겟 `NObeyesdad`의 직접 파생 변수(BMI = Weight/Height²)입니다.\n"
       "- 이 두 컬럼을 피처로 사용하면 모델이 사실상 BMI 공식을 재학습하는 것과 같아\n"
       "  99%+ 정확도가 나오지만 **실제 예측 능력은 없습니다**.\n"
       "- 재작성된 코드는 **행동·생활습관 피처만** 사용합니다."
       ),
    code(
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import pickle\n"
        "import os\n\n"
        "os.makedirs('../artifacts', exist_ok=True)\n\n"
        "DATA_PATH = '../data/obesity.csv'\n"
        "ARTIFACTS = '../artifacts'"
    ),
    code(
        "data = pd.read_csv(DATA_PATH)\n"
        "print(f'Shape: {data.shape}')\n"
        "print(f'Columns: {list(data.columns)}')\n"
        "data.head()"
    ),
    md("## 타겟 분포 확인"),
    code(
        "obesity_order = [\n"
        "    'Insufficient_Weight', 'Normal_Weight',\n"
        "    'Overweight_Level_I', 'Overweight_Level_II',\n"
        "    'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'\n"
        "]\n\n"
        "counts = data['NObeyesdad'].value_counts().reindex(obesity_order)\n"
        "plt.figure(figsize=(10, 4))\n"
        "sns.barplot(x=obesity_order, y=counts.values, palette='RdYlGn_r')\n"
        "plt.xticks(rotation=30, ha='right', fontsize=9)\n"
        "plt.title('Obesity Level Distribution')\n"
        "plt.ylabel('Count')\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "print(counts)"
    ),
    md("## 피처 선택\n\n"
       "행동·생활습관 피처만 사용하고 `Height`, `Weight`는 제외합니다."),
    code(
        "CATEGORICAL_FEATURES = [\n"
        "    'Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE',\n"
        "    'family_history_with_overweight', 'CAEC', 'MTRANS'\n"
        "]\n"
        "CONTINUOUS_FEATURES = ['Age', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']\n"
        "TARGET = 'NObeyesdad'\n\n"
        "print('피처 (behavioral only):')\n"
        "print(f'  범주형: {CATEGORICAL_FEATURES}')\n"
        "print(f'  연속형: {CONTINUOUS_FEATURES}')\n"
        "print(f'  제외 (data leakage): [Height, Weight]')"
    ),
    md("## EDA – 연속형 피처 분포"),
    code(
        "fig, axes = plt.subplots(2, 3, figsize=(14, 7))\n"
        "for ax, col in zip(axes.flatten(), CONTINUOUS_FEATURES):\n"
        "    sns.histplot(data[col], kde=True, ax=ax, bins=25, color='steelblue')\n"
        "    ax.set_title(col)\n"
        "plt.suptitle('Behavioral Continuous Features', fontsize=13, y=1.02)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md("## EDA – 범주형 피처 분포"),
    code(
        "fig, axes = plt.subplots(2, 4, figsize=(18, 8))\n"
        "for ax, col in zip(axes.flatten(), CATEGORICAL_FEATURES):\n"
        "    vc = data[col].value_counts()\n"
        "    sns.barplot(x=vc.index, y=vc.values, ax=ax, palette='Set2')\n"
        "    ax.set_title(col)\n"
        "    ax.tick_params(axis='x', rotation=30)\n"
        "plt.suptitle('Categorical Feature Distributions', fontsize=13, y=1.01)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md("## 범주형 인코딩\n\n"
       "- **이진 피처**: 딕셔너리 매핑으로 0/1\n"
       "- **순서형 피처**: 순서가 있는 카테고리를 정수로 (CALC, CAEC: no < Sometimes < Frequently < Always)\n"
       "- 인코더 딕셔너리를 `artifacts/encoders.pkl`에 저장 → 예측 시 동일 매핑 재사용"),
    code(
        "BINARY_MAP = {\n"
        "    'Gender': {'Female': 0, 'Male': 1},\n"
        "    'FAVC':   {'no': 0, 'yes': 1},\n"
        "    'SCC':    {'no': 0, 'yes': 1},\n"
        "    'SMOKE':  {'no': 0, 'yes': 1},\n"
        "    'family_history_with_overweight': {'no': 0, 'yes': 1},\n"
        "}\n"
        "ORDINAL_MAP = {\n"
        "    'CALC':   ['no', 'Sometimes', 'Frequently', 'Always'],\n"
        "    'CAEC':   ['no', 'Sometimes', 'Frequently', 'Always'],\n"
        "    'MTRANS': ['Bike', 'Walking', 'Public_Transportation', 'Motorbike', 'Automobile'],\n"
        "}\n\n"
        "df = data.copy()\n\n"
        "for col, mapping in BINARY_MAP.items():\n"
        "    df[col] = df[col].map(mapping)\n\n"
        "for col, order in ORDINAL_MAP.items():\n"
        "    order_map = {v: i for i, v in enumerate(order)}\n"
        "    df[col] = df[col].map(order_map)\n\n"
        "encoders = {'binary': BINARY_MAP, 'ordinal': ORDINAL_MAP}\n"
        "with open(f'{ARTIFACTS}/encoders.pkl', 'wb') as f:\n"
        "    pickle.dump(encoders, f)\n\n"
        "print('encoders.pkl 저장 완료')\n"
        "df[CATEGORICAL_FEATURES].head()"
    ),
    md("## 연속형 피처 스케일링\n\n"
       "StandardScaler를 피팅하고 `artifacts/scaler.pkl`에 저장합니다."),
    code(
        "from sklearn.preprocessing import StandardScaler\n\n"
        "scaler = StandardScaler()\n"
        "df[CONTINUOUS_FEATURES] = scaler.fit_transform(df[CONTINUOUS_FEATURES])\n\n"
        "with open(f'{ARTIFACTS}/scaler.pkl', 'wb') as f:\n"
        "    pickle.dump(scaler, f)\n\n"
        "print('scaler.pkl 저장 완료')\n"
        "print(f'스케일링 전 평균(원본): {data[CONTINUOUS_FEATURES].mean().round(2).to_dict()}')\n"
        "print(f'스케일링 후 평균:       {df[CONTINUOUS_FEATURES].mean().round(4).to_dict()}')"
    ),
    md("## 상관관계 분석"),
    code(
        "feature_df = df[CATEGORICAL_FEATURES + CONTINUOUS_FEATURES]\n"
        "corr = feature_df.corr()\n\n"
        "plt.figure(figsize=(10, 8))\n"
        "sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.4)\n"
        "plt.title('Feature Correlation Heatmap (Behavioral Features)')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md("## 전처리 완료 데이터 저장"),
    code(
        "save_cols = CATEGORICAL_FEATURES + CONTINUOUS_FEATURES + [TARGET]\n"
        "df[save_cols].to_csv(f'{ARTIFACTS}/processed_data.csv', index=False)\n\n"
        "print(f'processed_data.csv 저장 완료: {df.shape}')\n"
        "print(f'\\n피처 ({len(save_cols)-1}개) + 타겟 저장')\n"
        "df[save_cols].head()"
    ),
]

# ---------------------------------------------------------------------------
# Notebook 2 – Clustering
# ---------------------------------------------------------------------------
nb2_cells = [
    md("# 2. 비만 패턴 클러스터링 (Clustering)\n\n"
       "**목적:** 행동 피처만으로 유사한 비만 위험 프로파일을 가진 그룹을 찾습니다.\n\n"
       "**알고리즘:** KMedoids (실제 데이터 포인트를 중심으로 사용 → 해석 용이)\n\n"
       "**입력:** `artifacts/processed_data.csv` (1번 노트북 실행 후)\n"
       "**출력:** `artifacts/cluster_model.pkl`, `artifacts/cluster_summaries.json`, "
       "`artifacts/obesity_with_clusters.csv`"
       ),
    code(
        "import os\n"
        "os.environ['OPENBLAS_NUM_THREADS'] = '1'  # KMedoids 병렬 처리 경고 방지\n\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import pickle, json\n\n"
        "from sklearn_extra.cluster import KMedoids\n"
        "from sklearn.metrics import silhouette_score\n"
        "from sklearn.decomposition import PCA"
    ),
    code(
        "CATEGORICAL_FEATURES = [\n"
        "    'Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE',\n"
        "    'family_history_with_overweight', 'CAEC', 'MTRANS'\n"
        "]\n"
        "CONTINUOUS_FEATURES = ['Age', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']\n"
        "BEHAVIORAL_FEATURES = CATEGORICAL_FEATURES + CONTINUOUS_FEATURES\n"
        "ARTIFACTS = '../artifacts'\n\n"
        "df = pd.read_csv(f'{ARTIFACTS}/processed_data.csv')\n"
        "X = df[BEHAVIORAL_FEATURES].values\n\n"
        "print(f'Loaded {len(X)} samples, {len(BEHAVIORAL_FEATURES)} features')\n"
        "df.head()"
    ),
    md("## 최적 클러스터 수 선택 – 실루엣 분석"),
    code(
        "print('실루엣 분석 중... (k=4~10)')\n"
        "scores = []\n"
        "k_range = range(4, 11)\n\n"
        "for k in k_range:\n"
        "    km = KMedoids(n_clusters=k, random_state=42, max_iter=300)\n"
        "    labels = km.fit_predict(X)\n"
        "    s = silhouette_score(X, labels, sample_size=1000, random_state=42)\n"
        "    scores.append(s)\n"
        "    print(f'  k={k}: silhouette={s:.4f}')\n\n"
        "best_k = list(k_range)[int(np.argmax(scores))]\n"
        "print(f'\\n최적 k (실루엣 최대): {best_k}')\n\n"
        "plt.figure(figsize=(8, 4))\n"
        "plt.plot(list(k_range), scores, 'o-', color='steelblue', linewidth=2)\n"
        "plt.axvline(best_k, color='red', linestyle='--', label=f'Best k={best_k}')\n"
        "plt.xlabel('Number of Clusters (k)')\n"
        "plt.ylabel('Silhouette Score')\n"
        "plt.title('KMedoids Silhouette Analysis')\n"
        "plt.xticks(list(k_range))\n"
        "plt.legend()\n"
        "plt.grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md("## 최종 모델 학습"),
    code(
        "N_CLUSTERS = best_k  # 실루엣 분석 결과 활용\n"
        "print(f'최종 클러스터 수: {N_CLUSTERS}')\n\n"
        "kmedoids = KMedoids(n_clusters=N_CLUSTERS, random_state=42, max_iter=500)\n"
        "df['cluster'] = kmedoids.fit_predict(X)\n\n"
        "with open(f'{ARTIFACTS}/cluster_model.pkl', 'wb') as f:\n"
        "    pickle.dump({'model': kmedoids, 'n_clusters': N_CLUSTERS,\n"
        "                 'features': BEHAVIORAL_FEATURES}, f)\n\n"
        "print('cluster_model.pkl 저장 완료')\n"
        "print('\\n클러스터별 샘플 수:')\n"
        "print(df['cluster'].value_counts().sort_index())"
    ),
    md("## PCA 2D 시각화"),
    code(
        "pca = PCA(n_components=2)\n"
        "X_pca = pca.fit_transform(X)\n"
        "pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])\n"
        "pca_df['cluster'] = df['cluster'].values\n"
        "pca_df['NObeyesdad'] = df['NObeyesdad'].values\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n\n"
        "sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='cluster',\n"
        "                palette='tab10', s=30, alpha=0.7, ax=axes[0])\n"
        "axes[0].set_title('KMedoids Clusters (PCA 2D)')\n\n"
        "obesity_order = ['Insufficient_Weight','Normal_Weight','Overweight_Level_I',\n"
        "                 'Overweight_Level_II','Obesity_Type_I','Obesity_Type_II','Obesity_Type_III']\n"
        "sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='NObeyesdad',\n"
        "                hue_order=obesity_order, palette='RdYlGn_r', s=30, alpha=0.7, ax=axes[1])\n"
        "axes[1].set_title('True Obesity Labels (PCA 2D)')\n"
        "axes[1].legend(fontsize=7, loc='upper right')\n\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "print(f'PCA 설명 분산: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}')"
    ),
    md("## 클러스터별 통계 요약"),
    code(
        "# 원본 데이터로 읽기 쉬운 통계 계산\n"
        "raw = pd.read_csv('../data/obesity.csv')\n"
        "raw['cluster'] = df['cluster'].values\n"
        "raw['BMI'] = raw['Weight'] / raw['Height'] ** 2\n\n"
        "display_cols = ['Age', 'BMI', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']\n"
        "cluster_stats = raw.groupby('cluster')[display_cols].mean().round(2)\n"
        "cluster_stats['dominant_label'] = (\n"
        "    raw.groupby('cluster')['NObeyesdad']\n"
        "       .agg(lambda x: x.value_counts().index[0])\n"
        ")\n"
        "cluster_stats['count'] = raw.groupby('cluster').size()\n\n"
        "print('클러스터별 통계 (원본 스케일):')\n"
        "display(cluster_stats)"
    ),
    md("## 클러스터 요약문 생성 및 저장"),
    code(
        "summaries = []\n"
        "for c_id in sorted(raw['cluster'].unique()):\n"
        "    g = raw[raw['cluster'] == c_id]\n"
        "    n = len(g)\n"
        "    m = g[display_cols].mean().round(2)\n"
        "    dominant = g['NObeyesdad'].value_counts().index[0]\n"
        "    family_pct = (g['family_history_with_overweight'] == 'yes').mean() * 100\n\n"
        "    summary = (\n"
        "        f'Cluster {c_id} (n={n}): '\n"
        "        f'Avg Age {m[\"Age\"]:.0f}yr, '\n"
        "        f'Avg BMI {m[\"BMI\"]:.1f}, '\n"
        "        f'Vegetable freq {m[\"FCVC\"]:.2f}/3, '\n"
        "        f'Meals/day {m[\"NCP\"]:.2f}, '\n"
        "        f'Water {m[\"CH2O\"]:.2f}L/day, '\n"
        "        f'Physical activity {m[\"FAF\"]:.2f}/3, '\n"
        "        f'Tech use {m[\"TUE\"]:.2f}hr/day, '\n"
        "        f'Family history of overweight {family_pct:.0f}%. '\n"
        "        f'Most common obesity label: {dominant}.'\n"
        "    )\n"
        "    summaries.append({'cluster': int(c_id), 'n': n,\n"
        "                       'dominant_label': dominant, 'summary': summary})\n"
        "    print(summary)\n\n"
        "with open(f'{ARTIFACTS}/cluster_summaries.json', 'w', encoding='utf-8') as f:\n"
        "    json.dump(summaries, f, indent=2, ensure_ascii=False)\n"
        "print('\\ncluster_summaries.json 저장 완료')"
    ),
    code(
        "df.to_csv(f'{ARTIFACTS}/obesity_with_clusters.csv', index=False)\n"
        "print('obesity_with_clusters.csv 저장 완료')"
    ),
]

# ---------------------------------------------------------------------------
# Notebook 3 – Prediction
# ---------------------------------------------------------------------------
nb3_cells = [
    md("# 3. 비만 위험도 예측 모델 (Prediction)\n\n"
       "**목적:** 행동·생활습관 피처로 비만 여부(is_obese)를 이진 분류합니다.\n\n"
       "**중요:** `Height`, `Weight`, `BMI`를 피처에서 완전히 제외합니다.\n"
       "실제 의료 현장에서 '습관 정보만으로 비만 위험을 얼마나 예측할 수 있는가'를 평가합니다.\n"
       "이로 인해 정확도는 99%가 아닌 **현실적인 수치**가 나옵니다.\n\n"
       "**입력:** `artifacts/processed_data.csv`, `artifacts/encoders.pkl`, `artifacts/scaler.pkl`\n"
       "**출력:** `artifacts/prediction_model.pkl`"
       ),
    code(
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import pickle\n\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score\n"
        "from sklearn.metrics import (\n"
        "    accuracy_score, classification_report,\n"
        "    confusion_matrix, ConfusionMatrixDisplay,\n"
        "    roc_auc_score, roc_curve, auc\n"
        ")\n\n"
        "ARTIFACTS = '../artifacts'"
    ),
    code(
        "df = pd.read_csv(f'{ARTIFACTS}/processed_data.csv')\n\n"
        "CATEGORICAL_FEATURES = [\n"
        "    'Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE',\n"
        "    'family_history_with_overweight', 'CAEC', 'MTRANS'\n"
        "]\n"
        "CONTINUOUS_FEATURES = ['Age', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']\n"
        "BEHAVIORAL_FEATURES = CATEGORICAL_FEATURES + CONTINUOUS_FEATURES\n\n"
        "# 이진 타겟: 비만 3유형 → 1, 나머지 → 0\n"
        "obese_labels = ['Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']\n"
        "df['is_obese'] = df['NObeyesdad'].isin(obese_labels).astype(int)\n\n"
        "print('타겟 분포:')\n"
        "print(df['is_obese'].value_counts())\n"
        "print(f'비만 비율: {df[\"is_obese\"].mean():.2%}')"
    ),
    md("## 학습/테스트 분할"),
    code(
        "X = df[BEHAVIORAL_FEATURES]\n"
        "y = df['is_obese']\n\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.2, random_state=42, stratify=y\n"
        ")\n\n"
        "print(f'학습: {len(X_train)} / 테스트: {len(X_test)}')"
    ),
    md("## 모델 학습 (Logistic Regression)"),
    code(
        "clf = LogisticRegression(\n"
        "    max_iter=2000,\n"
        "    class_weight='balanced',  # 클래스 불균형 보정\n"
        "    random_state=42\n"
        ")\n"
        "clf.fit(X_train, y_train)\n\n"
        "# 5-fold 교차 검증\n"
        "cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc')\n"
        "print(f'5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')"
    ),
    md("## 성능 평가"),
    code(
        "y_pred = clf.predict(X_test)\n"
        "y_proba = clf.predict_proba(X_test)[:, 1]\n\n"
        "acc = accuracy_score(y_test, y_pred)\n"
        "roc_auc = roc_auc_score(y_test, y_proba)\n\n"
        "print(f'정확도 (Accuracy):  {acc:.4f}')\n"
        "print(f'ROC AUC:           {roc_auc:.4f}')\n"
        "print()\n"
        "print(classification_report(y_test, y_pred, target_names=['Not Obese', 'Obese']))"
    ),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n\n"
        "# Confusion Matrix\n"
        "cm = confusion_matrix(y_test, y_pred)\n"
        "ConfusionMatrixDisplay(cm, display_labels=['Not Obese', 'Obese']).plot(ax=axes[0], cmap='Blues')\n"
        "axes[0].set_title('Confusion Matrix')\n\n"
        "# ROC Curve\n"
        "fpr, tpr, _ = roc_curve(y_test, y_proba)\n"
        "axes[1].plot(fpr, tpr, 'darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')\n"
        "axes[1].plot([0,1],[0,1],'navy',linestyle='--')\n"
        "axes[1].set_xlabel('False Positive Rate')\n"
        "axes[1].set_ylabel('True Positive Rate')\n"
        "axes[1].set_title('ROC Curve')\n"
        "axes[1].legend()\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md("## 피처 중요도 (회귀 계수)"),
    code(
        "coef_df = pd.DataFrame({\n"
        "    'Feature': BEHAVIORAL_FEATURES,\n"
        "    'Coefficient': clf.coef_[0]\n"
        "}).sort_values('Coefficient', key=abs, ascending=False)\n\n"
        "plt.figure(figsize=(10, 5))\n"
        "colors = ['#e74c3c' if c > 0 else '#3498db' for c in coef_df['Coefficient']]\n"
        "sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette=colors)\n"
        "plt.axvline(0, color='black', linewidth=0.8)\n"
        "plt.title('Logistic Regression Feature Importance\\n(양수=비만 위험 증가, 음수=감소)')\n"
        "plt.tight_layout()\n"
        "plt.show()\n\n"
        "display(coef_df)"
    ),
    md("## 모델 저장"),
    code(
        "with open(f'{ARTIFACTS}/prediction_model.pkl', 'wb') as f:\n"
        "    pickle.dump({\n"
        "        'model': clf,\n"
        "        'features': BEHAVIORAL_FEATURES,\n"
        "        'obese_labels': obese_labels\n"
        "    }, f)\n"
        "print('prediction_model.pkl 저장 완료')"
    ),
    md("## 비만 위험도 예측 함수\n\n"
       "저장된 인코더, 스케일러를 재사용하여 새로운 입력값 예측"),
    code(
        "def predict_obesity_risk(user_input: dict) -> dict:\n"
        "    \"\"\"\n"
        "    Parameters\n"
        "    ----------\n"
        "    user_input : dict\n"
        "        행동·생활습관 정보 (Height, Weight 불필요)\n"
        "        필수 키: Age, Gender, CALC, FAVC, FCVC, NCP, SCC,\n"
        "                 SMOKE, CH2O, family_history_with_overweight,\n"
        "                 FAF, TUE, CAEC, MTRANS\n"
        "\n"
        "    Returns\n"
        "    -------\n"
        "    dict with risk_score (0~1) and label\n"
        "    \"\"\"\n"
        "    with open(f'{ARTIFACTS}/encoders.pkl', 'rb') as f:\n"
        "        encoders = pickle.load(f)\n"
        "    with open(f'{ARTIFACTS}/scaler.pkl', 'rb') as f:\n"
        "        scaler = pickle.load(f)\n"
        "    with open(f'{ARTIFACTS}/prediction_model.pkl', 'rb') as f:\n"
        "        saved = pickle.load(f)\n"
        "    model = saved['model']\n"
        "    features = saved['features']\n\n"
        "    row = dict(user_input)\n"
        "    for col, mapping in encoders['binary'].items():\n"
        "        if col in row:\n"
        "            row[col] = mapping[row[col]]\n"
        "    for col, order in encoders['ordinal'].items():\n"
        "        if col in row:\n"
        "            row[col] = {v: i for i, v in enumerate(order)}[row[col]]\n\n"
        "    cat_feats = ['Gender','CALC','FAVC','SCC','SMOKE',\n"
        "                 'family_history_with_overweight','CAEC','MTRANS']\n"
        "    cont_feats = ['Age','FCVC','NCP','CH2O','FAF','TUE']\n"
        "    cont_values = np.array([[row[c] for c in cont_feats]])\n"
        "    scaled_cont = scaler.transform(cont_values)[0]\n"
        "    scaled_row = {c: scaled_cont[i] for i, c in enumerate(cont_feats)}\n"
        "    scaled_row.update({c: row[c] for c in cat_feats})\n\n"
        "    X_new = pd.DataFrame([scaled_row])[features]\n"
        "    prob = model.predict_proba(X_new)[0][1]\n\n"
        "    if prob < 0.3:\n"
        "        label, icon = '저위험', '🟢'\n"
        "    elif prob < 0.6:\n"
        "        label, icon = '중위험', '🟠'\n"
        "    else:\n"
        "        label, icon = '고위험', '🔴'\n\n"
        "    return {'risk_score': round(float(prob), 4), 'label': label, 'icon': icon}\n\n\n"
        "# 예시 입력\n"
        "sample = {\n"
        "    'Age': 28, 'Gender': 'Male',\n"
        "    'CALC': 'Sometimes', 'FAVC': 'yes',\n"
        "    'FCVC': 1, 'NCP': 3, 'SCC': 'yes', 'SMOKE': 'no',\n"
        "    'CH2O': 1, 'family_history_with_overweight': 'yes',\n"
        "    'FAF': 0, 'TUE': 2, 'CAEC': 'Frequently', 'MTRANS': 'Automobile'\n"
        "}\n\n"
        "result = predict_obesity_risk(sample)\n"
        "print(f'{result[\"icon\"]} 비만 위험도: {result[\"label\"]} (score: {result[\"risk_score\"]})')"
    ),
]

# ---------------------------------------------------------------------------
# Notebook 4 – RAG + LLM
# ---------------------------------------------------------------------------
nb4_cells = [
    md("# 4. RAG + LLM 비만 관리 가이드라인 추천\n\n"
       "**파이프라인:**\n"
       "1. `data/guidelines/` 내 TXT/PDF 파일을 문장 단위로 청크 분할\n"
       "2. SentenceTransformer로 청크 임베딩 생성 후 `artifacts/rag_chunks.npz`에 저장\n"
       "3. 클러스터 요약문(`artifacts/cluster_summaries.json`)을 쿼리로 사용\n"
       "4. 코사인 유사도로 관련 청크 Top-k 검색\n"
       "5. 검색된 청크 + 클러스터 프로파일을 Gemini에 전달 → 맞춤 추천 생성\n"
       "6. BERTScore로 답변 품질 평가\n\n"
       "**입력:** `data/guidelines/*.txt|.pdf`, `artifacts/cluster_summaries.json`\n"
       "**환경 변수:** `.env` 파일에 `GEMINI_API_KEY` 설정 필요"
       ),
    code(
        "%pip install -q sentence-transformers PyPDF2 nltk bert-score google-generativeai python-dotenv"
    ),
    code(
        "import os, re, json, glob\n"
        "import numpy as np\n"
        "import PyPDF2\n"
        "import nltk\n"
        "from nltk.tokenize import sent_tokenize\n"
        "from sentence_transformers import SentenceTransformer, util\n"
        "from dotenv import load_dotenv\n"
        "import google.generativeai as genai\n\n"
        "nltk.download('punkt', quiet=True)\n"
        "nltk.download('punkt_tab', quiet=True)\n\n"
        "load_dotenv('../.env')\n"
        "GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')\n"
        "if not GEMINI_API_KEY:\n"
        "    raise ValueError('GEMINI_API_KEY not found. Create .env from .env.example and set your key.')\n\n"
        "genai.configure(api_key=GEMINI_API_KEY)\n"
        "ARTIFACTS = '../artifacts'\n"
        "GUIDELINES_DIR = '../data/guidelines'"
    ),
    md("## 가이드라인 문서 로드 및 청크 분할\n\n"
       "문장 토크나이저로 분할한 후 window=3, stride=2로 슬라이딩 윈도우 청크를 만들면\n"
       "각 청크에 문맥이 충분히 포함됩니다."),
    code(
        "def load_text_from_file(path: str) -> str:\n"
        "    if path.endswith('.pdf'):\n"
        "        text = []\n"
        "        with open(path, 'rb') as f:\n"
        "            reader = PyPDF2.PdfReader(f)\n"
        "            for page in reader.pages:\n"
        "                t = page.extract_text()\n"
        "                if t:\n"
        "                    text.append(t)\n"
        "        return ' '.join(text)\n"
        "    with open(path, 'r', encoding='utf-8') as f:\n"
        "        return f.read()\n\n\n"
        "def make_chunks(text: str, window: int = 3, stride: int = 2) -> list[str]:\n"
        "    \"\"\"Sliding-window sentence-level chunking.\"\"\"\n"
        "    sentences = [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 20]\n"
        "    chunks = []\n"
        "    for i in range(0, len(sentences), stride):\n"
        "        chunk = ' '.join(sentences[i:i + window])\n"
        "        if chunk:\n"
        "            chunks.append(chunk)\n"
        "    return chunks\n\n\n"
        "all_chunks, all_sources = [], []\n"
        "files = glob.glob(f'{GUIDELINES_DIR}/*.txt') + glob.glob(f'{GUIDELINES_DIR}/*.pdf')\n\n"
        "if not files:\n"
        "    raise FileNotFoundError(f'No guideline files found in {GUIDELINES_DIR}')\n\n"
        "for fp in files:\n"
        "    doc_name = os.path.basename(fp)\n"
        "    text = load_text_from_file(fp)\n"
        "    chunks = make_chunks(text)\n"
        "    all_chunks.extend(chunks)\n"
        "    all_sources.extend([doc_name] * len(chunks))\n"
        "    print(f'  {doc_name}: {len(chunks)} chunks')\n\n"
        "print(f'\\n총 청크 수: {len(all_chunks)}')\n"
        "print('\\n샘플 청크 (처음 3개):')\n"
        "for i, c in enumerate(all_chunks[:3]):\n"
        "    print(f'  [{i}] {c[:150]}...')"
    ),
    md("## 청크 임베딩 생성 및 저장"),
    code(
        "print('SentenceTransformer 로딩 중...')\n"
        "embed_model = SentenceTransformer('all-MiniLM-L6-v2')\n\n"
        "print(f'임베딩 생성 중... ({len(all_chunks)}개 청크)')\n"
        "chunk_embeddings = embed_model.encode(\n"
        "    all_chunks,\n"
        "    batch_size=64,\n"
        "    show_progress_bar=True,\n"
        "    normalize_embeddings=True  # 코사인 유사도 계산을 내적으로 가속\n"
        ")\n\n"
        "np.savez(\n"
        "    f'{ARTIFACTS}/rag_chunks.npz',\n"
        "    embeddings=chunk_embeddings,\n"
        "    chunks=np.array(all_chunks),\n"
        "    sources=np.array(all_sources)\n"
        ")\n"
        "print(f'rag_chunks.npz 저장 완료: {chunk_embeddings.shape}')"
    ),
    md("## 클러스터 요약문 로드"),
    code(
        "with open(f'{ARTIFACTS}/cluster_summaries.json', encoding='utf-8') as f:\n"
        "    cluster_summaries = json.load(f)\n\n"
        "print(f'{len(cluster_summaries)}개 클러스터 요약문 로드 완료')\n"
        "for s in cluster_summaries[:2]:\n"
        "    print(f\"  Cluster {s['cluster']}: {s['summary'][:100]}...\")"
    ),
    md("## RAG 검색 함수"),
    code(
        "def retrieve_chunks(query: str, top_k: int = 5) -> list[dict]:\n"
        "    \"\"\"\n"
        "    Returns top-k guideline chunks most relevant to the query.\n"
        "    Assumes rag_chunks.npz is pre-loaded into module scope.\n"
        "    \"\"\"\n"
        "    data_npz = np.load(f'{ARTIFACTS}/rag_chunks.npz', allow_pickle=True)\n"
        "    embs = data_npz['embeddings']                  # (N, 384)\n"
        "    texts = data_npz['chunks'].tolist()\n"
        "    sources = data_npz['sources'].tolist()\n\n"
        "    q_emb = embed_model.encode([query], normalize_embeddings=True)[0]\n"
        "    scores = embs @ q_emb                          # dot product = cosine (normalized)\n"
        "    top_idx = np.argsort(scores)[::-1][:top_k]\n\n"
        "    return [\n"
        "        {'chunk': texts[i], 'source': sources[i], 'score': float(scores[i])}\n"
        "        for i in top_idx\n"
        "    ]"
    ),
    md("## 추천 생성 함수 (Gemini)"),
    code(
        "def generate_recommendation(cluster_summary: str, top_k: int = 5) -> dict:\n"
        "    \"\"\"Retrieve relevant guideline chunks and generate LLM recommendations.\"\"\"\n"
        "    retrieved = retrieve_chunks(cluster_summary, top_k=top_k)\n\n"
        "    context = '\\n\\n'.join(\n"
        "        f'[Source: {r[\"source\"]}] {r[\"chunk\"]}' for r in retrieved\n"
        "    )\n\n"
        "    prompt = (\n"
        "        'You are an expert obesity management clinician.\\n'\n"
        "        'Based on the patient profile and clinical guidelines below, '\n"
        "        'provide specific, actionable recommendations.\\n\\n'\n"
        "        f'## Patient Group Profile\\n{cluster_summary}\\n\\n'\n"
        "        f'## Relevant Guideline Excerpts\\n{context}\\n\\n'\n"
        "        '## Instructions\\n'\n"
        "        '1. Summarize key health risks for this group.\\n'\n"
        "        '2. Give 3-5 specific dietary recommendations.\\n'\n"
        "        '3. Give 2-3 physical activity recommendations.\\n'\n"
        "        '4. Note any behavioral interventions.\\n'\n"
        "        'Be concise and evidence-based.'\n"
        "    )\n\n"
        "    gemini = genai.GenerativeModel('gemini-1.5-flash')\n"
        "    response = gemini.generate_content(prompt)\n"
        "    return {'summary': cluster_summary, 'retrieved': retrieved, 'recommendation': response.text}\n\n\n"
        "# 단일 클러스터로 테스트\n"
        "test_cluster = cluster_summaries[0]\n"
        "print(f'테스트 클러스터: {test_cluster[\"summary\"][:120]}...')"
    ),
    code(
        "result = generate_recommendation(test_cluster['summary'], top_k=5)\n\n"
        "print('=' * 60)\n"
        "print(f'[Cluster {test_cluster[\"cluster\"]}] 추천 결과')\n"
        "print('=' * 60)\n"
        "print(result['recommendation'])\n"
        "print('\\n검색된 청크:')\n"
        "for i, r in enumerate(result['retrieved']):\n"
        "    print(f'  [{i+1}] (score={r[\"score\"]:.3f}) {r[\"chunk\"][:80]}...')"
    ),
    md("## 전체 클러스터 추천 생성"),
    code(
        "import time\n\n"
        "all_results = []\n"
        "for cs in cluster_summaries:\n"
        "    print(f'\\nCluster {cs[\"cluster\"]} ({cs[\"dominant_label\"]}) 처리 중...')\n"
        "    try:\n"
        "        res = generate_recommendation(cs['summary'], top_k=5)\n"
        "        all_results.append({'cluster': cs['cluster'],\n"
        "                            'label': cs['dominant_label'],\n"
        "                            'recommendation': res['recommendation']})\n"
        "        print(res['recommendation'][:300])\n"
        "    except Exception as e:\n"
        "        print(f'  오류: {e}')\n"
        "    time.sleep(1)  # API rate limit 방지\n\n"
        "print(f'\\n완료: {len(all_results)}/{len(cluster_summaries)} 클러스터')"
    ),
    md("## BERTScore 평가\n\n"
       "생성된 추천과 검색된 청크 간 의미론적 유사도를 BERTScore로 측정합니다."),
    code(
        "from bert_score import score as bert_score\n\n"
        "if all_results:\n"
        "    candidates = [r['recommendation'] for r in all_results]\n"
        "    # 각 클러스터의 검색된 청크 재활용 (참조)\n"
        "    references = [generate_recommendation(cluster_summaries[i]['summary'], top_k=3)['retrieved'][0]['chunk']\n"
        "                  for i in range(len(all_results))]\n\n"
        "    P, R, F1 = bert_score(\n"
        "        candidates, references,\n"
        "        lang='en', model_type='distilbert-base-uncased', verbose=False\n"
        "    )\n"
        "    print(f'BERTScore – Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}')\n"
        "    for i, (p, r, f) in enumerate(zip(P, R, F1)):\n"
        "        print(f'  Cluster {all_results[i][\"cluster\"]}: P={p:.3f} R={r:.3f} F1={f:.3f}')\n"
        "else:\n"
        "    print('추천 결과 없음 – 이전 셀을 먼저 실행하세요.')"
    ),
]

# ---------------------------------------------------------------------------
# Write notebooks
# ---------------------------------------------------------------------------
BASE = "/Users/cocoxoxo/rag-llm-obesity/code"
write(f"{BASE}/1_preprocessing.ipynb", nb(nb1_cells))
write(f"{BASE}/2_clustering.ipynb",    nb(nb2_cells))
write(f"{BASE}/3_prediction.ipynb",    nb(nb3_cells))
write(f"{BASE}/4_rag_llm.ipynb",       nb(nb4_cells))
print("\nAll notebooks created successfully.")
