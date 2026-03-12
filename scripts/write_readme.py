import sys
content = """# RAG + LLM 비만 관리 추천 시스템

행동/생활습관 데이터로 비만 위험을 분석하고, RAG와 Gemini LLM을 결합하여
개인화된 비만 관리 가이드라인을 추천하는 파이프라인입니다.

## 프로젝트 구조

```
rag-llm-obesity/
├── code/
│   ├── 1_preprocessing.ipynb   # EDA + 인코딩/스케일링 + 아티팩트 저장
│   ├── 2_clustering.ipynb      # KMedoids 클러스터링 + 실루엣 분석 + 요약문 생성
│   ├── 3_prediction.ipynb      # 이진 비만 예측 (행동 피처만) + 평가
│   └── 4_rag_llm.ipynb         # 가이드라인 RAG + Gemini 추천 + BERTScore
├── data/
│   ├── obesity.csv             # 원본 데이터셋 (2,111 samples, 17 features)
│   └── guidelines/
│       └── obesity_management_guidelines.txt
├── artifacts/                  # 실행 시 자동 생성 (gitignore)
├── .env.example
└── requirements.txt
```

## 설계 원칙 — 데이터 누수 방지

원본 코드의 핵심 문제: Height, Weight는 타겟 NObeyesdad의 직접 파생 변수입니다.

```
BMI = Weight / Height²  →  NObeyesdad (비만 분류)
```

이를 피처로 사용하면 모델이 사실상 BMI 공식을 재학습하여 정확도 99%+가 나오지만
**실제 예측 능력은 없습니다**. 재작성된 코드는 **행동/생활습관 피처 14개만** 사용합니다.

| 피처 | 설명 |
|------|------|
| Age | 나이 |
| Gender | 성별 |
| CALC | 알코올 섭취 빈도 |
| FAVC | 고열량 음식 자주 섭취 여부 |
| FCVC | 야채 섭취 빈도 (1-3) |
| NCP | 하루 식사 횟수 |
| SCC | 칼로리 모니터링 여부 |
| SMOKE | 흡연 여부 |
| CH2O | 하루 물 섭취량 (L) |
| family_history_with_overweight | 가족 과체중 이력 |
| FAF | 신체 활동 빈도 (0-3) |
| TUE | 기기 사용 시간 (hr/day) |
| CAEC | 식사 외 간식 섭취 빈도 |
| MTRANS | 주요 이동 수단 |

## 실행 방법

### 1. 환경 설정

```bash
pip install -r requirements.txt
cp .env.example .env
# .env 에 GEMINI_API_KEY 입력
```

### 2. 노트북 순서대로 실행

노트북은 **반드시 순서대로** 실행해야 합니다.

```
1_preprocessing   → artifacts/encoders.pkl, scaler.pkl, processed_data.csv
2_clustering      → artifacts/cluster_model.pkl, cluster_summaries.json
3_prediction      → artifacts/prediction_model.pkl
4_rag_llm         → 클러스터별 맞춤 가이드라인 추천 (Gemini API 필요)
```

## 기술 스택

- **클러스터링**: KMedoids (scikit-learn-extra) + 실루엣 분석으로 최적 k 선택
- **예측**: Logistic Regression (행동 피처만, class_weight=balanced)
- **임베딩**: SentenceTransformer all-MiniLM-L6-v2
- **LLM**: Google Gemini 1.5 Flash
- **RAG 검색**: 코사인 유사도 (정규화 임베딩 내적)
- **평가**: BERTScore (distilbert-base-uncased)

## 데이터셋

- **출처**: [UCI Obesity Dataset](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)
- 샘플 수: 2,111 / 피처: 17 / 타겟: 7-class 비만 분류
- 참고: 77%는 합성 데이터(SMOTE)로 생성된 것으로 알려져 있음
"""
with open("/Users/cocoxoxo/rag-llm-obesity/README.md", "w", encoding="utf-8") as f:
    f.write(content)
print("README.md written")
