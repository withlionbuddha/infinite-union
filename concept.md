# Modality Concept

## 1. Modality란 무엇인가?

**Modality**는 Machine Learning과 Deep Learning에서 **model이 처리하는 data의 형태 또는 유형**을 의미합니다.

동일한 대상이나 현상을 관찰하더라도 data가 표현되거나 측정되는 방식이 다르면 서로 다른 modality로 구분할 수 있습니다.

대표적인 modality는 아래와 같습니다.

```text
현실의 대상 또는 현상
        │
        ├── 사진 또는 영상       → Image Modality
        ├── 문장 또는 문서       → Text Modality
        ├── 음성 또는 소리       → Audio Modality
        └── 시간에 따른 측정값   → Time-series Modality
```

예를 들어 사람의 상태를 분석하는 model에서는 얼굴 사진을 Image Modality, 증상에 대한 문장을 Text Modality, 음성을 Audio Modality, 시간에 따라 측정한 심박수와 같은 data를 Time-series Modality로 다룰 수 있습니다.

---

## 2. Modality와 Features의 차이

Modality와 features는 서로 다른 개념입니다.

- **Modality**는 model에 입력되는 data의 형태 또는 유형입니다.
- **Features**는 encoder가 해당 modality의 input을 처리하여 추출하거나 학습한 representation입니다.

```text
Modality                 Encoder                 Features

Image       ─────→    Image Encoder    ─────→ Image Features
Text        ─────→    Text Encoder     ─────→ Text Features
Audio       ─────→    Audio Encoder    ─────→ Audio Features
Time-series ─────→ Time-series Encoder ─────→ Time-series Features
```

따라서 Image 자체와 Image Features는 동일한 것이 아닙니다. Image는 input data의 modality이고, Image Features는 Image Encoder가 Image input으로부터 생성한 feature representation입니다.

이와 같은 features를 **modality-specific features**라고 표현할 수 있습니다.

---

## 3. Single Modality와 Multiple Modalities

하나의 modality만 사용하는 model은 single-modality model로 볼 수 있습니다.

```text
Image
  ↓
Image Encoder
  ↓
Image Features
  ↓
Prediction
```

둘 이상의 서로 다른 modality를 하나의 target task에서 사용하는 경우 Multimodal Learning을 적용할 수 있습니다.

```text
Image ───────→ Image Encoder ───────→ Image Features ───────┐
                                                            │
Text ────────→ Text Encoder ────────→ Text Features ────────┼→ Interaction → Prediction
                                                            │
Time-series ─→ Time-series Encoder ─→ Time-series Features ─┘
```

여러 종류의 data가 존재한다는 사실만으로 Multimodal Learning을 적용하는 것은 아닙니다. 각각의 modality-specific features가 target task에 필요한지 판단하고, 여러 modality를 함께 사용했을 때 model performance가 개선되는지 검증해야 합니다.

---

## 4. Modality와 Interaction

Multimodal Learning에서 서로 다른 modality를 사용한다고 해서 modality 사이의 관계가 자동으로 학습되는 것은 아닙니다.

각 modality의 encoder는 먼저 modality-specific features를 생성할 수 있으며, 서로 다른 modality-specific features 사이의 관계 학습이 필요한 경우 별도의 Interaction mechanism을 적용할 수 있습니다.

```text
Image Modality
      ↓
Image Encoder
      ↓
Image Features ──────────┐
                         │
                         ├→ Interaction → Prediction
                         │
Text Features ───────────┘
      ↑
Text Encoder
      ↑
Text Modality
```

Interaction mechanism에는 Attention, Cross-Attention, Gating, Similarity, Feature Selection 등의 방법을 적용할 수 있습니다.

Attention은 modality 자체를 의미하지 않습니다. Attention은 features 또는 representation 사이의 관계와 relevance를 계산하기 위한 mechanism입니다.

---

## 5. Transformer Attention과 Modality

Transformer의 Self-Attention은 동일한 representation sequence 내부의 요소 사이 관계를 계산하는 데 사용할 수 있습니다.

예를 들어 Text Transformer에서는 token representations 사이의 관계를 계산할 수 있습니다.

```text
Text Modality
     ↓
Token Representations
     ↓
Self-Attention
     ↓
Text Features
```

Multimodal model에서는 서로 다른 modality-specific features 사이의 관계를 계산하기 위해 Cross-Attention과 같은 mechanism을 적용할 수 있습니다.

```text
Text Features  → Query
Image Features → Key
Image Features → Value
        │
        ↓
Cross-Attention
        │
        ↓
Cross-modal Interaction Features
```

따라서 **Attention은 관계를 계산하는 mechanism이고, Modality는 model이 처리하는 data의 형태 또는 유형**이라는 점을 구분해야 합니다.

---

## 6. Infinite Union에서의 Modality

Infinite Union에서는 각 modality와 modality-specific features의 identity와 independence를 유지하는 것을 기본 원칙으로 합니다.

```text
Image       → Image Encoder       → Image Features
Text        → Text Encoder        → Text Features
Audio       → Audio Encoder       → Audio Features
Time-series → Time-series Encoder → Time-series Features
                                      │
                                      ↓
                                Features Union
```

Features Union에 여러 modality-specific features가 포함되더라도 각 features가 어떤 modality에서 생성되었는지 식별하고 개별적으로 접근할 수 있어야 합니다.

필요한 경우 Interaction component가 Features Union에서 특정 modality-specific features를 선택하여 서로 연결하고 관계를 계산합니다.

이를 통해 Infinite Union은 **각 features의 독립성을 유지하면서 연결하고 상호작용하며, 새로운 modality와 features를 지속적으로 확장할 수 있는 modeling methodology**를 구성하는 것을 목표로 합니다.
