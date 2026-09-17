# AI 적용 여부 판단 기준

AI를 적용할 수 있다는 것과 **AI를 적용하는 것이 적절하다는 것**은 다릅니다.

AI 적용 여부를 결정하기 전에는 최소한 아래 다섯 가지 항목을 확인하는 것이 좋습니다.

1. Data Availability
2. Evaluation Metric
3. Inference Cost
4. Explainability
5. Maintenance Cost

이 문서는 각 항목을 실제 프로젝트에서 쉽게 판단할 수 있도록 질문 중심으로 정리합니다.

---

## 1. Data Availability

**Data Availability**는 AI model이 학습하고 검증하는 데 필요한 data를 실제로 확보할 수 있는지를 의미합니다.

단순히 data의 개수가 많은지만 확인하는 것이 아니라 **수량, 품질, 정답 정보, 대표성, 사용 가능 여부**를 함께 확인해야 합니다.

### 확인할 질문

- 해결하려는 문제와 직접 관련된 data가 존재하는가?
- model을 학습할 만큼 충분한 양을 확보할 수 있는가?
- classification 문제라면 정답 label을 확보할 수 있는가?
- 잘못된 값, 누락된 값, 중복 data가 지나치게 많지 않은가?
- 실제 운영 환경에서 발생하는 다양한 상황이 data에 포함되어 있는가?
- training data와 validation data 또는 test data를 분리할 수 있는가?
- 개인정보, 저작권, license 등의 조건상 학습에 사용할 수 있는 data인가?
- 운영 이후에도 새로운 data를 지속적으로 수집할 수 있는가?

### 쉬운 예

제품 불량 판정 AI를 만든다고 가정합니다.

```text
제품 이미지 100,000장 있음
        ↓
불량 여부 label이 없음
        ↓
Supervised Learning에 바로 사용하기 어려움
```

반대로 아래와 같은 data가 있다면 적용 가능성을 더 구체적으로 검토할 수 있습니다.

```text
정상 제품 이미지
불량 제품 이미지
불량 유형 label
다양한 조명과 촬영 조건
충분한 학습 sample
        ↓
Training / Validation / Test Dataset 구성 가능
```

### 핵심 판단

**"AI가 학습해야 할 pattern을 보여주는 충분하고 신뢰할 수 있는 data가 있는가?"**

이 질문에 답하기 어렵다면 model architecture를 선택하기 전에 data 확보 방법부터 검토해야 합니다.

---

## 2. Evaluation Metric

**Evaluation Metric**은 AI model이 문제를 얼마나 잘 해결했는지를 숫자로 측정하는 기준입니다.

AI를 적용하기 전에 **무엇을 성공으로 볼 것인지 측정할 수 있어야 합니다.**

### 확인할 질문

- model의 성공과 실패를 측정할 수 있는가?
- 실제 업무 목적과 evaluation metric이 연결되어 있는가?
- baseline과 AI model을 같은 조건에서 비교할 수 있는가?
- 운영에 필요한 최소 성능 기준을 정의할 수 있는가?
- 특정 오류가 다른 오류보다 더 중요한 문제인가?

### 문제에 따른 대표적인 Evaluation Metric

| 문제 | 사용할 수 있는 Evaluation Metric |
| --- | --- |
| Classification | Accuracy, Precision, Recall, F1 Score |
| Regression | Mean Absolute Error, Mean Squared Error, Root Mean Squared Error |
| Object Detection | Mean Average Precision |
| Ranking / Retrieval | Precision at K, Recall at K, Mean Reciprocal Rank |
| Language Model | Cross-Entropy Loss, Perplexity, task-specific evaluation |

Evaluation Metric은 문제에 따라 선택해야 합니다.

예를 들어 불량품을 정상으로 잘못 판단하는 것이 큰 손실을 발생시키는 제조 검사에서는 단순 Accuracy만 확인하는 것보다 **불량품을 얼마나 놓치지 않는지 측정하는 Recall**이 중요할 수 있습니다.

### 핵심 판단

**"AI가 좋아졌다고 말할 수 있는 객관적인 측정 기준이 있는가?"**

측정 기준을 정의할 수 없다면 model 개선 여부와 실제 적용 효과를 판단하기 어렵습니다.

---

## 3. Inference Cost

**Inference Cost**는 학습이 완료된 AI model이 실제 input을 받아 prediction을 생성할 때 필요한 시간과 computing resource 비용을 의미합니다.

높은 model performance만으로 실제 서비스에 적용할 수 있는 것은 아닙니다.

### 확인할 질문

- prediction 결과가 몇 millisecond 또는 몇 second 안에 나와야 하는가?
- Central Processing Unit만으로 실행할 수 있는가?
- Graphics Processing Unit 또는 별도의 accelerator가 필요한가?
- 동시에 몇 개의 요청을 처리해야 하는가?
- model을 cloud에서 실행할 것인가, edge device에서 실행할 것인가?
- 필요한 memory를 실제 운영 장치가 제공할 수 있는가?
- 요청량이 증가했을 때 computing cost가 감당 가능한 수준인가?

### 쉬운 예

```text
Model A
Accuracy: 95%
Inference Time: 30 milliseconds

Model B
Accuracy: 96%
Inference Time: 2 seconds
```

Model B의 Accuracy가 더 높더라도 100 milliseconds 이내의 응답이 필요한 real-time system이라면 그대로 적용하기 어려울 수 있습니다.

따라서 model performance와 inference requirement를 함께 검증해야 합니다.

### 핵심 판단

**"필요한 응답 시간과 hardware 조건에서 이 model을 실제로 운영할 수 있는가?"**

---

## 4. Explainability

**Explainability**는 AI model이 특정 prediction을 생성한 이유를 사람이 어느 정도 이해하고 분석할 수 있는지를 의미합니다.

모든 AI system에 동일한 수준의 explainability가 필요한 것은 아닙니다. 그러나 prediction 결과가 중요한 의사결정에 사용될수록 model의 판단 근거를 확인할 필요성이 커질 수 있습니다.

### 확인할 질문

- prediction 결과만 있으면 되는가, 판단 근거도 필요한가?
- 잘못된 prediction이 발생했을 때 원인을 분석할 수 있어야 하는가?
- 사용자 또는 담당자에게 결과의 근거를 설명해야 하는가?
- 어떤 input 또는 features가 prediction에 영향을 주었는지 분석할 필요가 있는가?
- model 내부의 weights, gradients, activations 또는 attention behavior를 진단해야 하는가?

### 쉬운 예

추천 시스템에서는 추천 결과 자체가 주요 목적일 수 있습니다.

반면 제조 품질 검사에서는 다음과 같은 질문이 중요할 수 있습니다.

```text
왜 이 제품을 불량으로 판단했는가?
        ↓
어떤 영역 또는 features가 판단에 영향을 주었는가?
        ↓
model 오류인가?
data 문제인가?
촬영 조건 문제인가?
```

Explainability가 필요한 문제라면 model performance뿐 아니라 prediction을 분석할 수 있는 방법도 architecture와 함께 설계해야 합니다.

### 핵심 판단

**"AI의 결과뿐 아니라 결과가 만들어진 이유도 확인해야 하는 문제인가?"**

---

## 5. Maintenance Cost

**Maintenance Cost**는 AI model을 한 번 개발하는 비용이 아니라 운영 이후 지속적으로 유지하는 데 필요한 비용을 의미합니다.

일반 software와 달리 AI system은 source code뿐 아니라 **data, model, preprocessing, evaluation, deployment, monitoring**도 함께 관리해야 합니다.

### 확인할 질문

- 새로운 data가 들어오면 model을 다시 학습해야 하는가?
- data distribution이 시간에 따라 변할 가능성이 있는가?
- model performance를 운영 환경에서 지속적으로 측정할 수 있는가?
- model version과 dataset version을 관리할 수 있는가?
- 재학습과 validation을 자동화할 필요가 있는가?
- model serving infrastructure를 운영할 인력과 비용이 있는가?
- 장애가 발생했을 때 기존 rule 또는 이전 model로 되돌릴 수 있는가?

### 쉬운 예

```text
일반 Software
Source Code 변경
    ↓
Test
    ↓
Deploy
```

AI system에서는 관리 대상이 더 많아질 수 있습니다.

```text
Data 변경
    ↓
Preprocessing
    ↓
Model Training
    ↓
Evaluation
    ↓
Model Versioning
    ↓
Deployment
    ↓
Monitoring
    ↓
필요한 경우 Retraining
```

따라서 개발 당시 높은 성능이 나온다는 이유만으로 AI 적용을 결정해서는 안 됩니다.

### 핵심 판단

**"이 model을 개발한 뒤에도 지속적으로 운영하고 검증할 수 있는가?"**

---

## 6. 빠르게 판단하는 Checklist

아래 질문은 초기 단계에서 AI 적용 가능성을 빠르게 확인하기 위한 checklist입니다.

| 판단 항목 | 핵심 질문 |
| --- | --- |
| Data Availability | 학습과 검증에 필요한 충분하고 신뢰할 수 있는 data가 있는가? |
| Evaluation Metric | AI의 성공과 실패를 객관적인 숫자로 측정할 수 있는가? |
| Inference Cost | 필요한 응답 시간과 hardware 조건에서 운영할 수 있는가? |
| Explainability | prediction 결과의 근거를 필요한 수준까지 분석할 수 있는가? |
| Maintenance Cost | data와 model을 지속적으로 monitoring, validation, retraining할 수 있는가? |

판단 흐름을 단순화하면 아래와 같습니다.

```text
해결하려는 문제 정의
        ↓
학습 가능한 Data가 있는가?
        │
        ├── No → Data 확보 또는 Rule-based Approach 검토
        │
        └── Yes
             ↓
성능을 측정할 Evaluation Metric이 있는가?
        │
        ├── No → 성공 기준부터 정의
        │
        └── Yes
             ↓
운영 환경에서 Inference가 가능한가?
        │
        ├── No → Model 경량화 또는 다른 Approach 검토
        │
        └── Yes
             ↓
필요한 Explainability를 제공할 수 있는가?
        │
        ├── No → 분석 방법 또는 다른 Approach 검토
        │
        └── Yes
             ↓
지속적인 Maintenance가 가능한가?
        │
        ├── No → 운영 구조와 비용 재검토
        │
        └── Yes
             ↓
AI 적용 후보
             ↓
Baseline과 비교 실험
             ↓
실제 AI 적용 여부 결정
```

이 checklist의 모든 항목을 단순히 Yes 또는 No로 평가하는 것이 목적은 아닙니다. 각 항목의 요구 수준은 target task와 실제 운영 환경에 따라 달라집니다.

최종적으로는 **AI model이 기존 software algorithm, rule-based system 또는 baseline보다 실제 문제 해결에 충분한 효과를 제공하는지 동일한 evaluation condition에서 검증한 후 적용 여부를 결정해야 합니다.**
