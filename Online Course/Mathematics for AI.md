- [Lecture](https://youtube.com/playlist?list=PLP37CSD8dyIZUftqc3ZFSaCyPe1ZepPXL&si=cYHWed5n1sYVhzJZ ) #lecture/NBC #DL/RNN #DL/CNN #ML/MLE
## Deep Learning 작동 원리 이해하기
### Neural Network and MLP
- Neural Network : linear model과 activation function이 합성된 함수
- MLP : Neural Network가 multi-layer로 합성된 함수
  ![|600](https://i.imgur.com/V2mUG5Z.png)
- 층이 깊어질수록, 목적함수를 projection하는데 필요한 node(neuron)의 수가 훨씬 빨리 줄어들어 좀 더 효율적인 학습이 가능함

### Backpropagation Algorithm
![|600](https://i.imgur.com/UW5SFec.png)
- ‘Output -> Input’ 로 계산해가며 parameter를 학습함.
- chain-rule 기반의 자동 미분을 사용한다.

## Basic of Probability Theory
### 왜 확률론을 배워야 하는가?
- DL확률론 기반의 ML theory에 바탕을 두고 있음.
- Loss function의 작동 원리는 data의 통계적으로 해석에 기반을 둠.
- R) Goal of L2-노름(유클리드, loss function in regression model) : min(Variance)
- R) Goal of cross-entropy : min(model prediction에서 불확실성)
- 따라서 Var, 불확실성을 계산하기 위해 확률론을 알아야 함.

### 확률변수와 확률분포
![|300](https://i.imgur.com/813o4vK.png)

- 확률 변수는 확률분포에 따라 Discrete(이산형), Continuous(연속형)으로 구분
- Discrete variable’s modeling : Calculate all possible case
- Continuous variable’s modeling : Calculate density in data space using integral
- 조건부확률분포 P(x|y)는 data space에서 x(input)과 y(output) 사이의 관계를 modeling

### 조건부 확률과 기계학습
- Logistic Regression에서 사용했던 linear model과 softmax function의 결합을 통해 data에서 추출된 pattern 기반의 확률 해석이 가능하다.
- Classification에서는 x(data)에서 추출된 특징패턴과 W(가중치행렬)을 이용해 조건부 확률을 계산
- Regression에서는 조건부 기대값인 E[y|x]을 추정한다.
- DL은 다층신경망을 통해 data로부터 특징 패턴을 추출함
$$
\mathbb{E}_{\mathbf{x}\sim P(\mathbf{x})}[f(\mathbf{x})] = \int_{\mathcal{X}} f(\mathbf{x})P(\mathbf{x})d\mathbf{x}, \quad \mathbb{E}_{\mathbf{x}\sim P(\mathbf{x})}[f(\mathbf{x})] = \sum_{\mathbf{x}\in\mathcal{X}} f(\mathbf{x})P(\mathbf{x})
$$

### Monte Carlo Sampling
$$
\mathbb{E}_{\mathbf{x}\sim P(\mathbf{x})}[f(\mathbf{x})] \approx \frac{1}{N}\sum_{i=1}^N f(\mathbf{x}^{(i)})
$$
- ML의 많은 문제들은 확률분포를 명시적으로 모르는 경우가 많다.
- 독립추출만 보장된다면, 대수(large number)의 법칙에 의해 수렴성을 보장함
- 불확실성의 추정, model 평가 및 최적화에 매우 유용해 다양하게 이용된다.
- 대표적인 응용 분야 : BNNs, RL, Hyperparameter Optimization

## Basic of Statistics
### 모수적(parametric) 방법론
- Goal of statistical modeling : 확률분포를 추정(inference, 매우 다양함)
- 그러나 유한한 개수의 data를 관찰해 모집단의 분포를 정확하게 알아내는 것은 불가능하다. -> 근사적인(approximate) 확률분포를 추정할 수 밖에 없음
- 모수적 방법론 : data가 특정 확률분포를 따른다고 선험적으로(a priori) 가정한 후, 그 분포를 결정하는 모수(parameter)를 추정하는 방법
- 비모수적 방법론 : 특정 확률분포를 가정하지 않으며, data에 따라 model의 구조 및 모수의 개수를 유연하게 조정하면서 확률분포를 추정함.
- 표집 분포(Sampling distribution) : 표본통계량이 이론적으로 따르는 확률분포

### 최대가능도 추정법(Maximum Likelihood Estimation, MLE)
- 표본평균, 표본분산은 분포마다 모수가 다르므로 때로는 적절한 통계량이 아니다.
- Goal : 이론적으로, 가장 가능성(likelihood)이 높은 모수를 추정하는 것
- 모수 최적화, data의 확률 분포 학습, 과적합 방지, loss function 정의 등 매우 다양한 분야에서 이용된다.
- Argmax : 주어진 함수의 값을 최대화하는 모수를 찾는 연산.
- x(data set)이 독립적으로 추출된 경우, 로그가능도(log-likelihood)를 최적화함.
$$
\hat{\theta}_{\text{MLE}} = \underset{\theta}{\text{argmax }} L(\theta; \mathbf{x}) = \underset{\theta}{\text{argmax }} P(\mathbf{x}|\theta), \quad \hat{\theta}_{\text{MLE}} = \underset{\theta}{\text{argmax }} \frac{1}{n}\sum_{i=1}^n\sum_{k=1}^K y_{i,k}\log(\text{MLP}_{\theta}(\mathbf{x}_{i,k}))
$$

### 왜 로그가능도를 이용하는가?
- N(data)가 충분한 대수(large-number)가 된다면, 컴퓨터의 정확도로 가능도를 계산하는 것이 불가능해진다.
- 경사하강법으로 가능도를 최적화할 때 미분 연산을 이용하는데, 로그가능도를 이용하면 연산량이 O(n**2)에서 O(n)으로 줄어든다.

### 확률분포의 거리 구하기
- ML에서 사용되는 손실함수들은 P(사전분포), Q(사후분포)를 통해 유도한다.
- TV distance, KLD, Wasserstein Distance는 대표적인 방법들
- KLD는 P 분포와 Q분포가 얼마나 다른지 측정하는 방법
- MLE에서 KLD는 Divergence(발산)을 최소화하는 방향으로 학습한다.
$$
D_{\text{KL}}(P \| Q) = \sum_{\mathbf{x}\in\mathcal{X}} P(\mathbf{x})\log\left(\frac{P(\mathbf{x})}{Q(\mathbf{x})}\right)
$$
## Basic of Deep Bayesian Statistics
### 조건부 확률과 베이즈 정리
![](https://i.imgur.com/mwxzsVt.png)
- R) If ‘P(A | B)=posterior’, then ‘P(A)=prior’, ‘P(B)=evidence’, ‘P(B | A)=likelihood’
- 베이즈 정리 핵심 : 새로운 data가 추가되었을 때, 앞서 계산한 posterior을 prior로 이용해 갱신된 사후확률을 계산할 수 있다.
- 조건부 확률은 유용한 통계적 해석을 제공하지만, 인과관계(강력한 예측)를 지닌다고 판단하기는 어렵다.

## Basic of RNN : 그림으로 이해하기
### Convolution 연산 이해하기
- MLP와 다르게, kernel을 위치가 고정된 Input(Vector) 상에서 움직여가며 선형모델, 합성함수 적용한다.
- kernel은 모든 Input에 공통적으로 적용되므로 역전파를 계산할 때에도 convolution 연산이 수행한다.
- Convolution 연산은 Image, 영상 처리에서 매우 유용하게 쓰이므로, 작동 원리를 정확히 이해하는 것이 중요.
![](https://i.imgur.com/Ulc6Np2.png)


## Basic of RNN
### Sequence data
- 소리, 문자열, 주가 등 불연속적인(시간, 발생 순서 중요) 사건으로 이루어진 data.
- 독립동등분포(i.i.d) : dataset의 sample들이 서로 독립적(Independent)이고, Identically Distributed(동일한 확률분포)를 따른다는 의미.
- P) Sequence data는 시간적, 순차적 연속성을 지니는 경우가 많아서 i.i.d 가정을 위배하는 경우가 많다.
- S) 조건부 확률 : 이전 sequence info를 이용해 미래의 data 확률분포 예측에 효과적.
- 길이가 가변적인 data를 다룰 수 있는 model이 필요.
- S) RNN : 입력 data의 순서를 고려하여, 이전 시점의 정보를 현재 시점으로 전달하는 구조. 조건부 확률을 이용해 반복 학습함.
![|600](https://i.imgur.com/V9QR2Hm.png)

### Problem of Vanilla RNN : Vanishing and Exploding Gradients
- RNN은 BPTT를 이용해 미분의 곱으로 이루어진 가중치행렬의 미분(1)을 계산한다.
- Vanishing Gradient : Sequence의 길이가 길어진다면, (1)의 계산이 초반의 정보를 잃는 문제가 발생해, 장기 의존성을 학습하는데 어려움이 생긴다.
- Exploding Gradient : Vanishing Gradient와 반대로, Gradient가 매우 커져서 학습이 불안해지는 문제