# infinite-union

`infinite-union`은 서로 다른 modality에서 추출된 **modality-specific features의 identity와 independence를 유지하면서**, 필요한 interaction을 명시적으로 구성하고 새로운 modality 또는 model component를 **확장할 수 있도록 설계하는 multimodal architecture 관점**입니다.

Infinite Union은 multimodal fusion에서 발생할 수 있는 modality-specific feature identity 저하, component 간 결합도 증가, 확장성 및 observability 저하와 같은 **구조적 한계를 극복하기 위해 제안하는 multimodal architecture 관점**입니다.

> **Infinite Union**은 표준 ML/DL 용어가 아니라 이 repository에서 제안하는 **modeling methodology**로, 각각의 features가 독립성을 유지하면서 서로 연결되고 상호작용하며, 새로운 features와 model components로 지속적으로 확장할 수 있도록 모델을 구성하는 방법입니다.

---

## 1. AI가 적합한 문제와 적합하지 않은 문제

Deep Learning은 입력과 출력 사이의 관계를 고정된 규칙만으로 명확하게 기술하기 어렵고, 충분한 data에서 representation 또는 pattern을 학습해야 하는 문제에 적합합니다.

대표적인 예는 아래와 같습니다.

- Image classification / object detection
- Natural Language Processing
- Speech / audio recognition
- Time-series forecasting / classification
- Anomaly detection
- Multimodal learning

반면에, 명확한 business rule, 단순 lookup, sorting, deterministic calculation처럼 입력과 출력 관계를 규칙으로 정확하게 정의할 수 있는 문제는 일반적인 software algorithm이나 rule-based system이 더 단순하고 검증하기 쉬울 수 있습니다. (AI가 만사가 아니라고! 반도체 값만 올리는 주범)

AI 적용 여부는 data availability, evaluation metric, inference cost, explainability, maintenance cost를 기준으로 판단하고 결정해야 합니다.

---

## 2. 주요 Data Modalities

이 repository에서 기본적으로 다루는 modality는 아래와 같습니다.

```text
Image       → Image Encoder       → Image Features
Text        → Text Encoder        → Text Features
Audio       → Audio Encoder       → Audio Features
Time-series → Time-series Encoder → Time-series Features
```

각 encoder는 자신의 modality에 적합한 representation을 학습합니다.

`Image Features`, `Text Features`, `Audio Features`, `Time-series Features`는 서로 다른 encoder가 생성한 **modality-specific features**입니다.

---

## 3. Pretrained Model

Pretrained model은 대규모 dataset으로 먼저 학습된 model을 downstream task에 활용하는 방식입니다.

주요 장점은 feature representation을 처음부터 학습하는 비용을 줄일 수 있고, 제한된 downstream dataset에서도 transfer learning을 적용할 수 있다는 점입니다.

Pretrained model의 적용 여부는 pretrained data와 target domain 사이의 distribution 차이, model size, inference cost, fine-tuning cost, license 등의 조건을 확인한 후 결정해야 합니다.

Infinite Union에서는 각 modality encoder를 독립적인 component로 취급하므로 pretrained encoder를 modality별로 선택하거나 교체할 수 있도록 설계합니다.

---

## 4. Multimodal Learning

Multimodal learning은 둘 이상의 modality를 함께 사용하여 task를 수행하는 machine learning approach입니다.

예시는 아래와 같습니다.

```text
Image + Text
Audio + Text
Image + Time-series
Image + Text + Audio + Time-series
```

한 modality에서 부족한 feature representation을 다른 modality가 보완할 수 있으며, 서로 다른 modality 사이의 관계를 학습할 수 있습니다.

반면 modality별 preprocessing과 encoder가 필요하고, representation dimension과 temporal/spatial alignment가 다를 수 있으며, missing modality와 training/inference cost를 추가로 처리해야 합니다.

---

## 5. Pretrained Model과 Multimodal Learning 적용 결정 기준

Pretrained Model과 Multimodal Learning은 항상 적용하는 것이 아니라 **target task, dataset, modality, model performance 및 compute cost를 기준으로 적용 여부를 판단하고 결정해야 합니다.**

| 판단 기준 | Pretrained Model | Multimodal Learning |
| --- | --- | --- |
| Dataset 규모 | 학습 data가 부족하고 target task에 적합한 pretrained representation이 존재하는지 판단합니다. | 각 modality의 학습에 필요한 data와 paired/aligned data가 확보되는지 판단합니다. |
| 기존 model 활용 | target task 또는 target domain에 적용 가능한 pretrained model을 선택할 수 있는지 확인합니다. | 필요한 modality별 encoder를 선택하거나 직접 학습해야 하는지 결정합니다. |
| 입력 modality | single modality 또는 multiple modalities 모두에 적용할 수 있습니다. | 둘 이상의 modality-specific features가 target task에 필요한 경우 적용 여부를 판단합니다. |
| Feature 필요성 | pretrained representation을 재사용하는 것이 task-specific training보다 효과적인지 검증합니다. | 서로 다른 modality-specific features가 task 수행에 필요한 추가 정보를 제공하는지 검증합니다. |
| Domain 차이 | pretraining domain과 target domain의 차이를 확인하고 fine-tuning 또는 task-specific model 적용 여부를 결정합니다. | modality별 data distribution과 alignment 조건을 확인하고 interaction 방식의 적용 여부를 결정합니다. |
| Compute Cost | training 및 inference resource 요구량을 측정하여 model size와 적용 방식을 선택합니다. | modality별 encoder와 interaction component에 필요한 training/inference resource를 측정하여 적용 여부를 결정합니다. |
| 구현 복잡도 | feature extraction, fine-tuning, full training 중 적용 방식을 선택합니다. | preprocessing, alignment, missing modality 처리 및 interaction mechanism의 구현 범위를 결정합니다. |
| 성능 검증 | scratch model 또는 baseline과 동일한 evaluation condition에서 비교합니다. | single-modality baseline 및 modality ablation 결과와 동일한 evaluation condition에서 비교합니다. |

핵심 판단 과정은 아래와 같습니다.

```text
Target Task
    │
    ↓
필요한 modality를 판단
    │
    ├── Single Modality
    │       │
    │       ↓
    │   적용 가능한 Pretrained Model이 있는가?
    │       ├── Yes → Pretrained Model 적용 여부 판단
    │       └── No  → Task-specific Model 적용
    │
    └── Multiple Modalities
            │
            ↓
    각 modality-specific features가
    target task에 필요한가?
            │
       ┌────┴────┐
       │         │
      No        Yes
       │         │
       ↓         ↓
 Single       Multimodal Learning
 Modality     적용 여부 판단
                 │
                 ↓
         Pretrained Encoder를
         modality별로 적용할 수 있는가?
                 │
           ┌─────┴─────┐
          Yes          No
           │            │
           ↓            ↓
      Pretrained     Task-specific
      Encoders       Encoders
```

Multimodal Learning 적용 여부는 단순히 여러 종류의 data가 존재한다는 이유만으로 결정하지 않습니다. 여러 modality의 **modality-specific features가 target task에 필요한지**, 그리고 이들 사이의 interaction을 학습했을 때 single-modality baseline보다 model performance가 실제로 개선되는지를 실험으로 검증해야 합니다.

Multimodal architecture의 필요성은 single-modality baseline과 동일한 evaluation condition에서 비교하여 판단합니다. 각 modality-specific features를 추가하거나 제거하는 **Ablation Study**를 통해 해당 modality와 interaction component가 model performance에 미치는 영향을 분석할 수 있습니다.

---

## 6. Representative Multimodal Architectures

Multimodal architecture는 task와 interaction 시점에 따라 여러 방식으로 구성할 수 있습니다. 아래 분류는 대표적인 설계 예이며 고정된 taxonomy를 의미하지 않습니다.

### Early Combination

낮은 수준의 input 또는 representation을 비교적 이른 단계에서 함께 처리합니다.

```text
Input A ─┐
         ├→ Shared Processing → Prediction
Input B ─┘
```

### Late Combination

각 modality를 독립적으로 처리한 후 downstream stage에서 결과를 함께 사용합니다.

```text
Input A → Model A → Output A ─┐
                              ├→ Decision
Input B → Model B → Output B ─┘
```

### Intermediate Interaction

각 modality의 encoder가 생성한 intermediate features 사이의 관계를 model 내부에서 계산합니다.

```text
Input A → Encoder A → Features A ─┐
                                  ├→ Interaction → Prediction
Input B → Encoder B → Features B ─┘
```

---

## 7. Multimodal Combination에서 발생할 수 있는 구조적 한계

Multimodal model에서 여러 modality representation을 하나의 tensor 또는 shared representation으로 결합하는 설계는 효과적일 수 있습니다. 그러나 구현 방식에 따라 아래와 같은 문제가 발생할 수 있습니다.

- modality-specific feature identity 추적이 어려워질 수 있습니다.
- 특정 modality encoder를 독립적으로 교체하기 어려워질 수 있습니다.
- modality별 activation 또는 gradient diagnostics가 복잡해질 수 있습니다.
- 새로운 modality를 추가할 때 downstream component 변경 범위가 커질 수 있습니다.
- missing modality와 modality-specific ablation을 처리하기 어려울 수 있습니다.

이러한 문제는 모든 multimodal architecture에 항상 발생하는 것은 아니며 architecture와 implementation에 따라 달라집니다.

Infinite Union은 이러한 설계 문제를 줄이기 위해 **feature management와 cross-modal interaction의 책임을 분리**하는 multimodal architecture를 제안합니다.

---

## 8. Infinite Union

Infinite Union의 핵심은 서로 다른 modality-specific features를 하나의 representation으로 즉시 변환하는 것이 아니라, **각 features의 identity를 유지한 상태로 함께 관리하는 multimodal architecture**를 구성하는 것입니다.

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

`Features Union`은 이 repository에서 사용하는 multimodal architecture concept입니다. Features Union에서는 각 features의 독립성을 유지하고 개별적으로 식별·접근할 수 있도록 구성합니다. 예를 들어 Python에서는 아래와 같이 `dictionary` 객체를 사용하여 표현할 수 있습니다.

```python
features_union = {
    "image": image_features,
    "text": text_features,
    "audio": audio_features,
    "time_series": time_series_features,
}
```

Python `dictionary` 객체 자체가 새로운 ML algorithm이라는 의미는 아닙니다. 핵심 design constraint는 각 modality-specific features가 **독립적으로 식별되고 접근 가능한 상태를 유지하는 것**입니다.

```text
Features Union
├── Image Features
├── Text Features
├── Audio Features
└── Time-series Features
```

각 features는 함께 관리되지만 자신의 modality identity를 잃지 않습니다.

---

## 9. Infinite Extension

Features Union은 특정 modality 개수에 architecture concept을 고정하지 않습니다.

```text
Features Union
├── Image Features
├── Text Features
├── Audio Features
├── Time-series Features
├── New Modality Features
└── ...
```

새로운 modality를 추가할 때 기존 modality-specific features를 변경하는 대신 새로운 encoder와 features를 독립적으로 추가합니다.

실제 확장 가능성은 downstream interaction interface, dimension compatibility, training data, compute resource 등의 조건에 따라 결정됩니다. `Infinite`는 물리적으로 무제한이라는 의미가 아니라 **확장 가능한 multimodal architecture principle**을 나타냅니다.

---

## 10. Features Union과 Interaction의 분리

Features Union의 책임은 **features를 독립적으로 유지하고 관리하는 것**입니다.

Cross-modal relationship learning이 필요한 경우 별도의 `Interaction` component가 Features Union에서 필요한 features를 선택하여 참조합니다.

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

Interaction은 하나의 algorithm으로 고정하지 않습니다. Target task와 features의 관계에 따라 아래 mechanism 중 필요한 방식을 선택합니다.

### Attention

Attention은 Query와 Key 사이의 relevance를 계산하고 그 결과를 이용해 Value를 가중합니다.

```text
Attention(Q, K, V) = softmax(QKᵀ / √dₖ)V
```

### Cross-Attention

서로 다른 representation source 사이의 관계를 계산할 수 있습니다.

예를 들어 Text Features를 Query, Image Features를 Key/Value로 적용할 수 있습니다.

```text
CrossAttention(
    Q_text,
    K_image,
    V_image
)
```

### Gating

특정 features 또는 modality의 영향도를 조절합니다.

```text
g = sigmoid(Wx + b)
weighted_features = g × features
```

### Similarity

두 feature representation 사이의 similarity를 계산합니다.

예를 들어 cosine similarity는 아래와 같습니다.

```text
similarity(a, b) = (a · b) / (||a|| ||b||)
```

서로 다른 encoder의 raw features가 자동으로 비교 가능한 것은 아닙니다. 필요한 경우 projection layer 또는 representation learning을 적용하여 compatible embedding space를 구성합니다.

### Feature Selection

Task 또는 input condition에 따라 필요한 modality-specific features를 선택합니다.

```text
Task A → Image Features + Text Features
Task B → Text Features + Time-series Features
Task C → Audio Features
```

Selection 방식은 task requirement에 따라 deterministic rule 또는 learned routing mechanism 중에서 결정합니다.

---

## 11. Ablation Study

**Ablation Study**는 model의 특정 component 또는 input을 제거하거나 변경하고, 동일한 evaluation condition에서 model performance가 어떻게 변화하는지 비교하는 experimental method입니다.

Infinite Union에서는 modality-specific features의 identity가 유지되므로 특정 modality의 features를 선택적으로 제외하는 실험을 구성하기 쉽습니다.

예를 들어 Image + Text model은 아래 조건을 비교할 수 있습니다.

```text
1. Image Features + Text Features
2. Image Features only
3. Text Features only
```

Architecture 관점에서는 아래와 같습니다.

```text
Image Features ─┐
                ├→ Interaction → Prediction
Text Features ──┘
```

각각의 modality-specific features를 제거한 condition과 전체 condition의 evaluation metric을 비교합니다.

```text
Image + Text
Image only
Text only
```

이를 통해 **특정 modality-specific features를 제거하거나 사용했을 때 model performance가 어떻게 변화하는지 분석**할 수 있습니다.

단, performance difference를 특정 modality의 독립적인 기여량으로 바로 해석해서는 안 됩니다. Modality-specific features 사이의 interaction effect가 존재할 수 있기 때문입니다.

보다 엄밀한 분석에서는 동일한 dataset split, preprocessing, training condition, evaluation metric을 적용하고 필요에 따라 repeated experiments, masking 또는 component-level ablation의 적용 여부를 결정합니다.

---

## 12. Diagnostics / Observability

Modality-specific features의 identity를 유지하면 modality별 model diagnostics를 수행하기 쉬워집니다.

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

이를 통해 아래 항목을 modality 단위로 분석할 수 있습니다.

- 특정 encoder의 gradient가 지나치게 작거나 큰가?
- 특정 modality의 activation distribution이 비정상적인가?
- 특정 encoder의 weights가 실제로 update되고 있는가?
- Interaction 이후 특정 modality-specific features가 거의 사용되지 않는가?

---

## 13. Architecture Goal

Infinite Union은 multimodal architecture 관점에서 아래 구조를 제안합니다.

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

핵심은 **features management와 cross-modal relationship computation을 분리하는 것**입니다.

---

## 14. Design Principles

1. **Independence**  
   각 modality encoder와 modality-specific features를 독립적으로 다룰 수 있어야 합니다.

2. **Identity Preservation**  
   Features Union에 포함된 features가 어떤 modality에서 생성되었는지 식별할 수 있어야 합니다.

3. **Explicit Interaction**  
   Cross-modal relationship은 Features Union 자체가 아니라 명시적인 Interaction component에서 계산합니다.

4. **Extensibility**  
   새로운 modality 또는 encoder를 추가할 때 기존 component의 변경을 최소화합니다.

5. **Observability**  
   Modality-specific weights, gradients, activations 및 interaction behavior를 분석할 수 있도록 구성합니다.

6. **Replaceability**  
   특정 modality encoder를 다른 architecture 또는 pretrained model로 교체할 수 있도록 interface를 분리합니다.

---

## 15. Relationship with deep-learning-core

`deep-learning-core`는 reusable PyTorch training infrastructure를 담당하고, `infinite-union`은 modality-specific features와 cross-modal interaction의 구조를 정의하는 multimodal architecture 관점을 담당합니다.

```text
                    infinite-union
                          │
             Multimodal Architecture
                          │
       ┌──────────────────┼──────────────────┐
       ↓                  ↓                  ↓
    Encoders         Features Union      Interaction
       │                                      │
       └──────────────────┬───────────────────┘
                          ↓
                   deep-learning-core
                          │
       ┌──────────────────┼──────────────────┐
       ↓                  ↓                  ↓
    Training          Evaluation         Diagnostics
```

`deep-learning-core`의 training runner는 single Tensor뿐 아니라 `dictionary`, `tuple`, `list`와 같은 nested multimodal input data structures를 device로 이동할 수 있도록 구성할 수 있습니다.

예시는 아래와 같습니다.

```python
x_batch = {
    "image": image_tensor,
    "text": text_tensor,
    "time_series": time_series_tensor,
}
```

이를 통해 `infinite-union`의 multimodal architecture와 reusable training infrastructure의 책임을 분리합니다.

---

## 16. Vision

`infinite-union`의 목표는 여러 modality를 단순히 하나의 representation으로 만드는 것 자체가 아닙니다.

**Multimodal architecture 관점에서 modality-specific features의 independence와 identity를 유지하면서 필요한 cross-modal interaction을 명시적으로 구성하고, 새로운 modality와 model component를 확장할 수 있는 architecture를 설계하고 검증하는 것**을 목표로 합니다.

```text
Independent Modalities
        ↓
Modality-specific Encoders
        ↓
Modality-specific Features
        ↓
Features Union
        ↓
Explicit Interaction
        ↓
Prediction
```
