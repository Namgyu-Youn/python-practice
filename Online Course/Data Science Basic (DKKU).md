- [Lecture](https://youtube.com/playlist?list=PLP37CSD8dyIZUftqc3ZFSaCyPe1ZepPXL&si=cYHWed5n1sYVhzJZ) #lecture #data-mining #DL/RNN #DL/CNN #ML/MLE
## Data Mining Methodology
### Statistics vs Data-Mining 
- **Statistics**
  - 현실 재현에 부적합 (가정을 전제로 함)
  - 알고리즘이 선형적인 구조
- **Data Mining**
  - 현실의 noise를 반영
  - 알고리즘이 비선형적이고 Robust (low 가정의 제약)
  - 기법 구분 기준: Data의 size, pattern-type, noise, 해당 기법의 기본 가정을 data가 만족하는지 여부

### Learning Types
#### Supervised Learning
- X(feature) -> Y(label)의 training data로 test-data(미래) 예측
- **Types**:
  - Regression: data type = Int (ex. 24.5, 31.2)
  - Classification: data type = Class (ex. A,B,C)

#### Unsupervised Learning
- Y(label)의 Training data 없이 미래 예측
- **Methods**: Dimension Reduction, Clustering, Recommendation

---

## Data Mining Process
### Overall Process
1. Biz 목적 설정 (사용자, 일회성)
2. Data 수집
3. Data 탐색 (변수 설정)
4. DM 문제 설정 (회귀, 분류, 군집)
5. Data division (학습, 검증, 평가 용도로 구분)
6. Select DM method (Decision Tree, Logistic Regression)
7. Data Analysis (Base model, Model tuning)
8. 결과 해석
9. 모델 적용

### Variable Types
- **Numeric** (수치형, int)
- **Categorical** (범주형, class)
  - Nominal: no-rank (ex. M/F)
  - Ordinal: rank (ex. A,B,C)
  - One-hot encoding: Class를 Dummy Variable로 변환하여 Numeric으로 처리

### Data Preprocessing
#### Outlier 처리 (Novelty detection)
- 극단적인 Value 제거가 데이터 전처리의 핵심

#### NA(결측치) 처리
- **Removal**: 적은 수의 record가 NA인 경우
- **Imputation**: Entity의 많은 곳에 NA가 있는 경우 (mean, median으로 대체)

#### Normalizing/Standardizing
- 목적: 단위 차이가 큰 변수들을 동일 척도로 변환
- **Methods**:
  - Standard scaling (z-score): (x-mean)/SD
  - Min-Max scaling: (x-min)/(max-min)

---
## Performance Evaluation
### Regression Model's Valuation
- **Naive Benchmark**: Y의 Average를 통한 단순 예측
- **RSS(잔차 제곱 합)**: 예측값과 실제 값 차이의 제곱합

### Classification Model's Valuation
- **Binary Classification**: 0(Negative/False) - 1(Positive/True)
- **Key Metrics**:
  - Accuracy: Model의 분류 정확도
  - Confusion Matrix Components:
    - FP (False Positive): False를 Positive로 잘못 분류
    - TN (True Negative)
    - TP (True Positive)
    - FN (False Negative)
  - Accuracy = (TN+TP) / (TN+TF+FP+FN)

### Advanced Metrics
- **Propensity**: 특정 class 소속 확률
- **Cutoff**: 분류 기준선
- **Precision(TPR)** = TP / (FP+TP)
  - Negative를 Positive로 잘못 판단하는 비용이 큰 경우
- **Recall(FPR)** = TP / (FN+TP)
  - Positive를 Negative로 잘못 판단하는 비용이 큰 경우
- **F1 Score**: Precision과 Recall의 조화평균

---
## Multiple Linear Regression
### Basic Concepts
- **목표**: 새로운 관측치에 대한 정확한 prediction
- **특징**: 
  - Linear relationship between variables (y=ax+b)
  - MLP는 Variable 갯수만 증가한 확장 모델

### Gradient Descent (경사 하강법)
- 목표 지점(m)으로 기울기를 하강시키며 접근
- Gradient 감소에 따라 하강 거리도 감소
- **Learning Rate**: 접근 가중치 (점의 이동 거리와 비례)

### Variable Selection Methods
- **Feedforward**: 변수를 하나씩 추가하며 성능 비교
- **Backward Elimination**: 전체에서 하나씩 제거하며 비교
- **Stepwise**: 변수 추가/제거를 동적으로 수행

---
## KNN (K-Nearest Neighbors)
### Characteristics
- New data 입력 시 K개의 가장 가까운 기존 data로 class 경계 결정
- **Types**:
  - Instance-based Learning: 개별 관측치 기반 예측
  - Memory-based Learning: 학습 data 메모리 저장 후 예측
  - Lazy Learning: test data 입력 시점에 작동

### Hyperparameters
- **K**: 탐색할 인접 data 수
- **Distance Measures**: 
  - 거리 측정 방법
  - Normalization 필요
  - Scaling으로 단위 차이 왜곡 보정
- **Fitting Issues**:
  - Overfitting: K가 매우 작을 때 발생
  - Underfitting: K가 매우 클 때 발생

---
## Naive Bayes
### Statistical Concepts
- Trial (독립 시행)
- Intersection of events (곱사건)
- Difference of events (차사건)
- Mutually exclusive events (배반사건)
- **Probability Types**:
  - Classical: P(A) = N(A) / N(total)
  - Axiomatic: 0~1 범위 전제
  - Conditional: 조건부 확률

### Bayes Theorem
- **목적**: 조건부 확률 P(Ai|B) 계산
  - 과거 불확실성 기반 미래 예측
  - 다양한 조건으로 조건부 확률 추정
  - Pros : 단순 모델, 효율적 계산, 우수한 분류 성과
  - Cons : 많은 관측치 필요, Laplace smoothing으로 보완

---
## Decision Tree
### Basic Concepts
- 의사결정 규칙의 tree 구조화
- **Types**:
  - Classification Tree: Purity 기반 평가
  - Regression Tree: RSS 최소화 목표
- **pros**:
	  -  다양한 Feature type 처리 가능
	  - Feature Normalization 불필요
	  - 해석 가능한 결과
	  - 빠른 인식 작업
- **단점**:
	- 2개 가지로 인한 Instability
- **해결책**: 
  - Random forest Algorithm
  - Ensemble 기법 (AdaBoost, GBM, XGBoost, LightGBM, Stacking)