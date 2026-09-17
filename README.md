# infinite-union

`infinite-union`은 서로 다른 modality에서 추출된 **modality-specific features의 identity와 independence를 유지하면서**, 필요한 interaction을 명시적으로 구성하고 새로운 modality 또는 model component를 확장할 수 있도록 설계하는 multimodal architecture project이다.

> **Infinite Union**은 표준 ML/DL 용어가 아니라 이 프로젝트에서 사용하는 architecture design concept이다.

---

## 1. AI가 적합한 문제와 적합하지 않은 문제

Deep Learning은 입력과 출력 사이의 관계를 고정된 규칙만으로 명확하게 기술하기 어렵고, 충분한 data에서 representation 또는 pattern을 학습해야 하는 문제에 적합하다.

대표적인 예는 다음과 같다.

- Image classification / object detection
- Natural Language Processing
- Speech / audio recognition
- Time-series forecasting / classification
- Anomaly detection
- Multimodal learning

반대로 명확한 business rule, 단순 lookup, sorting, deterministic calculation처럼 입력과 출력 관계를 규칙으로 정확하게 정의할 수 있는 문제는 일반적인 software algorithm이나 rule-based system이 더 단순하고 검증하기 쉬울 수 있다.

AI 사용 여부는 문제 자체뿐 아니라 data availability, evaluation metric, inference cost, explainability, maintenance cost를 함께 고려하여 결정해야 한다.

---

## 2. 주요 Data Modalities

이 프로젝트에서 기본적으로 다루는 modality는 다음과 같다.

```text
Image       → Image Encoder       → Image Features
Text        → Text Encoder        → Text Features
Audio       → Audio Encoder       → Audio Features
Time-series → Time-series Encoder → Time-series Features
```

각 encoder는 자신의 modality에 적합한 representation을 학습한다.

`Image Features`, `Text Features`, `Audio Features`, `Time-series Features`는 서로 다른 encoder가 생성한 **modality-specific features**이다.

---

## 3. Pretrained Model

Pretrained model은 대규모 dataset으로 먼저 학습된 model을 downstream task에 활용하는 방식이다.

주요 장점은 feature representation을 처음부터 학습하는 비용을 줄일 수 있고, 제한된 downstream dataset에서도 transfer learning을 활용할 수 있다는 점이다.

반면 pretrained data와 target domain 사이의 distribution 차이, model size, inference cost, fine-tuning cost, license 등의 제약을 고려해야 한다.

Infinite Union에서는 각 modality encoder를 독립적인 component로 취급하므로 pretrained encoder를 modality별로 선택하거나 교체할 수 있도록 하는 것을 지향한다.

---

## 4. Multimodal Learning

Multimodal learning은 둘 이상의 modality를 함께 사용하여 task를 수행하는 machine learning approach이다.

예:

```text
Image + Text
Audio + Text
Image + Time-series
Image + Text + Audio + Time-series
```

장점은 한 modality에서 부족한 feature representation을 다른 modality가 보완할 가능성이 있고, 서로 다른 modality 사이의 관계를 학습할 수 있다는 점이다.

반면 modality별 preprocessing과 encoder가 필요하고, representation dimension과 temporal/spatial alignment가 다를 수 있으며, missing modality와 training/inference cost를 추가로 처리해야 한다.

---

## 5. Representative Multimodal Architectures

Multimodal architecture는 task와 interaction 시점에 따라 여러 방식으로 구성할 수 있다. 다음 분류는 대표적인 설계 예이며 고정된 taxonomy를 의미하지 않는다.

### Early Combination

낮은 수준의 input 또는 representation을 비교적 이른 단계에서 함께 처리한다.

```text
Input A ─┐
         ├→ Shared Processing → Prediction
Input B ─┘
```

### Late Combination

각 modality를 독립적으로 처리한 후 downstream stage에서 결과를 함께 사용한다.

```text
Input A → Model A → Output A ─┐
                              ├→ Decision
Input B → Model B → Output B ─┘
```

### Intermediate Interaction

각 modality의 encoder가 생성한 intermediate features 사이의 관계를 model 내부에서 계산한다.

```text
Input A → Encoder A → Features A ─┐
                                  ├→ Interaction → Prediction
Input B → Encoder B → Features B ─┘
```

---

## 6. Multimodal Combination에서 발생할 수 있는 구조적 한계

Multimodal model에서 여러 modality representation을 하나의 tensor 또는 shared representation으로 결합하는 설계는 효과적일 수 있다. 그러나 구현 방식에 따라 다음과 같은 문제가 발생할 수 있다.

- modality-specific feature identity 추적이 어려워질 수 있다.
- 특정 modality encoder를 독립적으로 교체하기 어려워질 수 있다.
- modality별 activation 또는 gradient diagnostics가 복잡해질 수 있다.
- 새로운 modality를 추가할 때 downstream component 변경 범위가 커질 수 있다.
- missing modality와 modality-specific ablation을 처리하기 어려울 수 있다.

이러한 문제는 모든 multimodal architecture에 항상 발생하는 것은 아니며 architecture와 implementation에 따라 달라진다.

Infinite Union은 이러한 설계 문제를 줄이기 위해 **feature management와 cross-modal interaction의 책임을 분리**하는 것을 목표로 한다.

---

## 7. Infinite Union

Infinite Union의 핵심은 서로 다른 modality-specific features를 하나의 representation으로 즉시 변환하는 것이 아니라, **각 features의 identity를 유지한 상태로 함께 관리하는 것**이다.

```text
Independent Inputs
        │
        ├───────────────┬────────────────┬────────────────┐
        ↓               ↓                ↓                ↓
      Image            Text             Audio         Time-series
        ↓               ↓                ↓                ↓
 Image Encoder     Text Encoder     Audio Encoder   Time-series Encoder
        ↓               ↓                ↓                ↓
Image Features    Text Features    Audio Features  Time-series Features
        │               │                │                │
        └───────────────┴────────────────┴────────────────┘
                                ↓
                         Features Union
                                │
                     ┌──────────┴──────────┐
                     ↓                     ↓
                Diagnostics           Interaction
                                           │
                                           ↓
                                       Prediction
```

### Features Union

`Features Union`은 이 프로젝트에서 사용하는 architecture concept이다. 구현에서는 다음과 같은 structured container로 표현할 수 있다.

```python
features_union = {
    "image": image_features,
    "text": text_features,
    "audio": audio_features,
    "time_series": time_series_features,
}
```

중요한 것은 `dict` 자체가 새로운 ML algorithm이라는 의미가 아니다. 핵심 design constraint는 각 modality-specific features가 **독립적으로 식별되고 접근 가능한 상태를 유지하는 것**이다.

즉:

```text
Features Union
├── Image Features
├── Text Features
├── Audio Features
└── Time-series Features
```

각 features는 함께 관리되지만 자신의 modality identity를 잃지 않는다.

---

## 8. Infinite Extension

Features Union은 특정 modality 개수에 architecture concept을 고정하지 않는다.

```text
Features Union
├── Image Features
├── Text Features
├── Audio Features
├── Time-series Features
├── New Modality Features
└── ...
```

새로운 modality를 추가할 때 기존 modality-specific features를 변경하는 대신 새로운 encoder와 features를 독립적으로 추가하는 방향을 지향한다.

실제 확장 가능성은 downstream interaction interface, dimension compatibility, training data, compute resource 등의 영향을 받으므로 `Infinite`는 물리적으로 무제한이라는 의미가 아니라 **확장 가능한 architecture principle**을 나타낸다.

---

## 9. Features Union과 Interaction의 분리

Features Union의 책임은 **features를 독립적으로 유지하고 관리하는 것**이다.

Cross-modal relationship learning이 필요한 경우 별도의 `Interaction` component가 Features Union에서 필요한 features를 참조한다.

```text
Features Union
├── Image Features
├── Text Features
├── Audio Features
└── Time-series Features
        │
        ↓
   Interaction
        │
        ↓
   Prediction
```

Interaction은 하나의 algorithm으로 고정되지 않는다.

### Attention

Attention은 Query와 Key 사이의 relevance를 계산하고 그 결과를 이용해 Value를 가중한다.

```text
Attention(Q, K, V) = softmax(QKᵀ / √dₖ)V
```

### Cross-Attention

서로 다른 representation source 사이의 관계를 계산할 수 있다.

예를 들어 Text Features를 Query, Image Features를 Key/Value로 사용할 수 있다.

```text
CrossAttention(
    Q_text,
    K_image,
    V_image
)
```

### Gating

특정 features 또는 modality의 영향도를 조절할 수 있다.

```text
g = sigmoid(Wx + b)
weighted_features = g × features
```

### Similarity

두 feature representation 사이의 similarity를 계산할 수 있다.

예를 들어 cosine similarity는 다음과 같다.

```text
similarity(a, b) = (a · b) / (||a|| ||b||)
```

서로 다른 encoder의 raw features가 자동으로 비교 가능한 것은 아니다. 필요한 경우 projection layer 또는 representation learning을 통해 compatible embedding space를 구성해야 한다.

### Feature Selection

Task 또는 input condition에 따라 필요한 modality-specific features를 선택할 수 있다.

```text
Task A → Image Features + Text Features
Task B → Text Features + Time-series Features
Task C → Audio Features
```

Selection은 deterministic rule 또는 learned routing mechanism으로 구현할 수 있다.

---

## 10. Ablation Study

**Ablation Study**는 model의 특정 component 또는 input을 제거하거나 변경하고, 동일한 evaluation condition에서 model performance가 어떻게 변화하는지 비교하는 experimental method이다.

Infinite Union에서는 modality-specific features의 identity가 유지되므로 특정 modality의 features를 선택적으로 제외하는 실험을 구성하기 쉽다.

예를 들어 Image + Text model은 다음 조건을 비교할 수 있다.

```text
1. Image Features + Text Features
2. Image Features only
3. Text Features only
```

Architecture 관점에서는 다음과 같다.

```text
Image Features ─┐
                ├→ Interaction → Prediction
Text Features ──┘
```

그리고 각각의 modality-specific features를 제거한 condition과 전체 condition의 evaluation metric을 비교한다.

```text
Image + Text
Image only
Text only
```

이를 통해 **특정 modality-specific features를 제거하거나 사용했을 때 model performance가 어떻게 변화하는지 분석**할 수 있다.

단, performance difference를 특정 modality의 독립적인 기여량으로 바로 해석해서는 안 된다. Modality-specific features 사이의 interaction effect가 존재할 수 있기 때문이다.

보다 엄밀한 분석에서는 동일한 dataset split, preprocessing, training condition, evaluation metric을 유지하고 필요하면 repeated experiments, masking 또는 component-level ablation을 함께 수행한다.

---

## 11. Diagnostics / Observability

Modality-specific features의 identity를 유지하면 modality별 model diagnostics를 수행하기 쉬워진다.

```text
Image Encoder
├── Weight
├── Gradient
└── Activation

Text Encoder
├── Weight
├── Gradient
└── Activation

Audio Encoder
├── Weight
├── Gradient
└── Activation

Time-series Encoder
├── Weight
├── Gradient
└── Activation
```

이를 통해 다음과 같은 질문을 modality 단위로 분석할 수 있다.

- 특정 encoder의 gradient가 지나치게 작거나 큰가?
- 특정 modality의 activation distribution이 비정상적인가?
- 특정 encoder의 weights가 실제로 update되고 있는가?
- Interaction 이후 특정 modality-specific features가 거의 사용되지 않는가?

---

## 12. Architecture Goal

Infinite Union은 다음 구조를 지향한다.

```text
Modality-specific Components
            ↓
Independent Encoders
            ↓
Modality-specific Features
            ↓
Features Union
            │
            ├────────→ Diagnostics
            │
            ↓
Interaction
├── Attention
├── Cross-Attention
├── Gating
├── Similarity
└── Selection
            │
            ↓
Prediction
```

핵심은 **features management와 relationship computation을 분리하는 것**이다.

---

## 13. Design Principles

1. **Independence**  
   각 modality encoder와 modality-specific features를 독립적으로 다룰 수 있어야 한다.

2. **Identity Preservation**  
   Features Union에 포함된 features가 어떤 modality에서 생성되었는지 식별할 수 있어야 한다.

3. **Explicit Interaction**  
   Cross-modal relationship은 Features Union 자체가 아니라 명시적인 Interaction component에서 계산한다.

4. **Extensibility**  
   새로운 modality 또는 encoder를 추가할 때 기존 component의 변경을 최소화한다.

5. **Observability**  
   Modality-specific weights, gradients, activations 및 interaction behavior를 분석할 수 있도록 한다.

6. **Replaceability**  
   특정 modality encoder를 다른 architecture 또는 pretrained model로 교체할 수 있도록 interface를 분리한다.

---

## 14. Relationship with deep-learning-core

`deep-learning-core`는 reusable PyTorch training infrastructure를 담당하고, `infinite-union`은 modality-specific features와 multimodal interaction을 구성하는 architecture layer를 담당하는 방향으로 분리한다.

```text
infinite-union
│
├── Modality-specific Encoders
├── Features Union
├── Interaction
├── Selection
└── Prediction
        │
        ↓
deep-learning-core
├── Training
├── Evaluation
├── Device Management
├── Checkpointing
└── Diagnostics
```

이 분리는 model-specific architecture와 reusable training infrastructure의 responsibility를 분리하기 위한 것이다.

---

## 15. Vision

Infinite Union의 목표는 여러 modality를 단순히 하나의 representation으로 만드는 것이 아니다.

각 modality-specific features를 독립적으로 식별하고 분석할 수 있는 상태로 유지하면서, task가 필요로 하는 관계만 명시적인 Interaction mechanism을 통해 계산하는 architecture를 구축하는 것이다.

```text
Independent Features
        ↓
Features Union
        ↓
Explicit Interaction
        ↓
Prediction
```

이를 기반으로 modality 추가, encoder 교체, ablation study, model diagnostics 및 새로운 interaction mechanism 실험을 일관된 architecture 안에서 수행할 수 있도록 하는 것을 목표로 한다.
