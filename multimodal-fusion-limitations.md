# Multimodal Fusion의 구조적 한계와 문제점

Multimodal Fusion은 image, text, audio, time-series 등 서로 다른 modality에서 생성된 정보를 결합하여 하나의 prediction 또는 decision에 활용하는 일반적인 방법입니다.

Fusion 자체가 잘못된 방법이라는 의미는 아닙니다. Early Fusion, Feature-level Fusion, Late Fusion, Attention 기반 Fusion 등 다양한 구조가 존재하며 각 방식은 서로 다른 장점과 한계를 가집니다.

그러나 여러 modality 또는 feature를 하나의 Fusion 구조를 중심으로 결합하는 경우, Fusion 방법에 따라 다음과 같은 구조적 한계가 발생할 수 있습니다.

## 1. Early Fusion의 한계

Early Fusion은 서로 다른 modality의 raw input 또는 초기 단계 feature를 비교적 이른 시점에 결합합니다.

```text
Image ───────┐
Text ────────┼──→ Early Fusion → Shared Model → Prediction
TimeSeries ──┘
```

이 방식은 구조가 단순할 수 있지만 modality별 정보가 모델의 초기 단계부터 결합되므로 이후 representation에서 각 modality의 영향을 독립적으로 추적하기 어려워질 수 있습니다.

또한 input structure, sampling rate, sequence length, scale 또는 dimensionality가 서로 다른 modality를 결합하려면 사전에 alignment와 preprocessing이 필요할 수 있습니다.

주요 한계는 다음과 같습니다.

- modality-specific identity 저하 가능성
- heterogeneous input alignment 필요
- modality별 preprocessing dependency 증가
- 특정 modality의 noise가 초기 단계부터 다른 modality 처리에 영향을 줄 가능성
- modality별 독립적인 진단의 어려움
- 새로운 modality 추가 시 input 및 초기 network 구조 변경 가능성

## 2. Feature-level Fusion의 한계

Feature-level Fusion은 각각의 encoder가 생성한 feature를 결합한 뒤 shared representation을 생성합니다.

```text
Image Encoder ──→ image_features ──────┐
                                       │
Text Encoder ───→ text_features ───────┼─→ Concat/Add → Fusion Layer
                                       │                  ↓
Audio Encoder ──→ audio_features ──────┘           fused_features
```

예를 들어 concatenation을 사용할 수 있습니다.

```python
fused_features = torch.cat(
    [image_features, text_features, audio_features],
    dim=-1,
)

output = fusion_layer(fused_features)
```

`torch.cat()` 직후에는 각 feature의 위치를 알고 있으므로 modality를 구별할 수 있습니다. 그러나 이후 Linear Layer, Multilayer Perceptron 또는 다른 transformation을 통과하면 shared representation이 생성됩니다.

```text
image_features ─┐
text_features ──┼─→ Fusion → [f1, f2, f3, ..., fn]
audio_features ─┘
```

이 시점부터 특정 latent feature가 어느 modality에서 유래했는지 또는 여러 modality의 결합으로 생성되었는지를 직접 구별하기 어려워질 수 있습니다.

주요 한계는 다음과 같습니다.

- modality-specific feature identity 저하 가능성
- feature provenance 추적의 어려움
- 개별 modality representation에 대한 observability 감소 가능성
- fusion layer에 대한 component coupling
- fusion input dimension에 대한 구조적 dependency
- 새로운 modality 추가 시 fusion layer 변경 가능성
- 기존 checkpoint와의 architecture compatibility 문제

예를 들어 Image 256차원과 Text 256차원을 결합하면 Fusion input은 512차원입니다. 여기에 Time-series 128차원을 추가하면 Fusion input은 640차원이 됩니다.

```python
# Before
nn.Linear(512, 512)

# After adding Time-series
nn.Linear(640, 512)
```

따라서 새로운 modality의 추가가 기존 Fusion component의 변경으로 전파될 수 있습니다.

## 3. Late Fusion의 한계

Late Fusion은 각 modality를 비교적 독립적으로 처리한 후 prediction, probability, score 또는 decision 수준에서 결과를 결합합니다.

```text
Image Model ──→ Prediction ──┐
                             │
Text Model ───→ Prediction ──┼─→ Late Fusion → Final Prediction
                             │
Audio Model ──→ Prediction ──┘
```

이 방식은 modality-specific model의 독립성과 observability를 비교적 잘 유지할 수 있습니다. 그러나 modality 사이의 low-level 또는 intermediate-level interaction을 충분히 학습하지 못할 수 있습니다.

주요 한계는 다음과 같습니다.

- fine-grained cross-modal interaction 제한
- modality 사이의 intermediate representation 공유 제한
- 각 model을 별도로 유지해야 하는 비용
- 여러 prediction을 결합하기 위한 weighting 또는 calibration 필요
- 특정 modality의 confidence scale 차이에 따른 편향 가능성
- modality 증가에 따른 model 및 inference cost 증가

즉 Late Fusion은 identity 문제를 완화하지만 interaction capability와 계산 비용 측면의 다른 trade-off를 가집니다.

## 4. Attention 기반 Fusion의 한계

Attention 또는 Cross-Attention 기반 architecture는 modality-specific representation을 유지하면서 modality 사이의 관계를 명시적으로 계산할 수 있습니다.

```text
image_features ─────────────┐
                            ├─→ Cross-Attention
text_features ──────────────┘
                            ↓
                  image_text_features
```

따라서 단순 Feature-level Fusion에서 발생할 수 있는 feature identity와 observability 문제를 완화할 수 있습니다. 그러나 원본 representation을 보존한다고 해서 구조적 문제가 모두 제거되는 것은 아닙니다.

### 4.1 Interaction component 증가

Image와 Text만 존재하면 하나의 pairwise interaction을 정의할 수 있지만 Audio와 Time-series가 추가되면 가능한 관계가 증가합니다.

```text
Image ↔ Text
Image ↔ Audio
Image ↔ TimeSeries
Text  ↔ Audio
Text  ↔ TimeSeries
Audio ↔ TimeSeries
```

모든 pairwise interaction을 명시적으로 구성한다면 modality가 `M`개일 때 가능한 unordered pair의 수는 다음과 같습니다.

```text
M(M - 1) / 2
```

방향성을 가진 Cross-Attention을 각각 별도로 구성한다면 최대 `M(M - 1)`개의 directed interaction 관계를 고려해야 할 수 있습니다.

따라서 modality가 증가할수록 interaction graph의 복잡성이 증가할 수 있습니다.

### 4.2 Interaction identity 복잡성

Attention 기반 구조에서는 원본 feature identity를 보존할 수 있습니다.

```text
image_features
text_features
audio_features
```

그러나 interaction 결과까지 독립적으로 관리하면 다음과 같은 representation이 계속 추가될 수 있습니다.

```text
image_text_features
image_audio_features
text_audio_features
image_timeseries_features
text_timeseries_features
audio_timeseries_features
...
```

따라서 원본 modality의 identity 문제를 완화하는 대신 interaction representation의 identity와 lifecycle을 관리하는 문제가 발생할 수 있습니다.

## 5. Component Coupling의 이동

Attention을 사용한다고 해서 component dependency가 없어지는 것은 아닙니다.

```python
cross_attention(
    query=text_features,
    key=image_features,
    value=image_features,
)
```

이 interaction은 feature dimension, token structure, embedding dimension, projection dimension, mask structure, sequence length, normalization 등의 interface에 의존할 수 있습니다.

예를 들어 Image Encoder 출력이 다음과 같이 변경되면:

```text
[B, 196, 768]
        ↓
[B, 256, 1024]
```

projection layer, adapter 또는 attention interface를 변경해야 할 수 있습니다.

즉 coupling이 제거되는 것이 아니라 Fusion Layer coupling에서 Interaction Interface coupling으로 이동할 수 있습니다.

## 6. Observability와 Diagnosis Complexity

Attention 기반 architecture는 image features, text features, Query/Key/Value projection, attention scores, attention weights, attention output, residual output 등을 각각 관찰할 수 있도록 구성할 수 있습니다.

이는 observability 측면의 장점입니다. 그러나 관찰 가능한 정보가 많다는 것이 문제 진단이 단순하다는 의미는 아닙니다.

성능이 저하되면 다음과 같은 여러 상태를 확인해야 할 수 있습니다.

```text
Image Encoder?
Text Encoder?
Image representation?
Text representation?
Q projection?
K projection?
V projection?
Attention distribution?
Residual connection?
Normalization?
Interaction representation?
Prediction head?
```

따라서 다음 두 개념은 동일하지 않습니다.

```text
Higher Observability ≠ Lower Diagnosis Complexity
```

Observability는 향상되더라도 진단해야 하는 component와 state의 수가 증가할 수 있습니다.

## 7. Modality Dominance

모든 modality가 architecture에 존재한다고 해서 모든 modality가 prediction에 실질적으로 기여하는 것은 아닙니다.

```text
Modality 존재 ≠ Modality contribution 보장
```

학습 결과 특정 modality가 prediction을 지배하고 다른 modality의 contribution이 매우 작아질 수 있습니다.

이를 확인하려면 modality ablation, gradient analysis, activation analysis, attention distribution analysis, representation similarity analysis 및 modality별 prediction contribution 비교 등이 필요할 수 있습니다.

따라서 modality identity를 보존하는 것과 modality contribution을 보장하는 것은 별개의 문제입니다.

## 8. Missing Modality 문제

실제 환경에서는 모든 modality가 항상 존재한다고 보장하기 어렵습니다.

```text
Training:  Image + Text + Audio
Inference: Image + Text + Missing Audio
```

Fusion architecture가 여러 modality의 동시 존재에 강하게 의존한다면 missing modality로 인해 성능이 크게 변할 수 있습니다.

이를 처리하기 위해 masking, imputation, fallback path, modality dropout 또는 missing-modality training 등의 별도 전략이 필요할 수 있습니다.

## 9. Modality Quality Imbalance

각 modality의 품질은 항상 동일하지 않습니다.

```text
Image       → high quality
Text        → high quality
Audio       → noisy
TimeSeries  → partially missing
```

Fusion은 이러한 정보를 결합하므로 품질이 낮은 modality가 전체 representation 또는 prediction에 영향을 줄 수 있습니다.

따라서 modality의 존재 여부뿐 아니라 quality, reliability, confidence, noise 및 missingness를 함께 판단해야 할 수 있습니다.

## 10. Representation Alignment 문제

서로 다른 modality는 본질적으로 representation의 의미와 구조가 다릅니다.

```text
Image       [B, patches, embedding]
Text        [B, tokens, embedding]
Audio       [B, frames, embedding]
TimeSeries  [B, timesteps, features]
```

Dimension을 동일하게 projection한다고 해서 의미적으로 alignment되었다고 볼 수는 없습니다.

```text
Same Dimension ≠ Same Semantics
```

따라서 multimodal Fusion에서는 representation alignment 자체가 별도의 학습 문제가 될 수 있습니다.

## 11. Temporal 및 Spatial Alignment 문제

Video, audio, sensor, time-series가 함께 사용되는 경우 동일한 사건을 서로 다른 시간 단위 또는 공간 단위로 관찰할 수 있습니다.

```text
Video frame     t = 10.2 sec
Audio segment   t = 10.0 ~ 10.5 sec
Sensor sample   t = 10.17 sec
Text event      timestamp 불명확
```

Fusion 이전 또는 interaction 과정에서 어떤 정보가 서로 대응하는지 결정해야 하며, 잘못된 alignment는 올바르지 않은 cross-modal relationship을 학습하게 만들 수 있습니다.

## 12. Computational Cost 증가

Attention 기반 multimodal architecture에서는 modality별 encoder 계산뿐 아니라 projection과 interaction 계산도 추가됩니다.

```text
Encoder Cost
     +
Projection Cost
     +
Interaction Cost
     +
Fusion / Prediction Cost
```

특히 token 또는 sequence 수준의 Cross-Attention은 sequence length가 증가하면서 상당한 memory와 computation을 요구할 수 있습니다.

여러 modality pair에 별도의 interaction을 적용하면 parameters, memory, training time, inference latency 및 interaction computation이 증가할 수 있습니다.

## 13. 확장성 문제

새로운 Sensor modality를 기존 Image, Text, Audio 구조에 추가한다고 가정합니다.

Fusion 중심 architecture에서는 다음 중 여러 부분을 변경해야 할 수 있습니다.

```text
Fusion input
Projection layer
Attention layer
Interaction graph
Mask
Normalization
Prediction head
Training pipeline
Checkpoint
Evaluation
Diagnostics
```

따라서 modality를 추가할 수 있다는 것과 기존 component를 변경하지 않고 확장할 수 있다는 것은 다릅니다.

```text
Can Add a Modality ≠ Structurally Extensible
```

확장성을 판단할 때는 새로운 component의 추가 가능성뿐 아니라 기존 component에 발생하는 변경의 범위도 중요합니다.

## 14. Ablation 및 원인 분석 복잡성

Multimodal model의 성능 변화가 어떤 component에서 발생했는지 판단하려면 여러 조합을 비교해야 할 수 있습니다.

```text
Image
Text
Audio
Image + Text
Image + Audio
Text + Audio
Image + Text + Audio
```

Interaction component까지 존재한다면 실험 조합은 더 증가할 수 있습니다.

```text
Image + Text without interaction
Image + Text with Cross-Attention
Image + Audio with Cross-Attention
...
```

따라서 modality와 interaction이 증가할수록 모델 성능의 원인을 검증하기 위한 Ablation Study 비용도 증가할 수 있습니다.

## 15. 학습 Dependency와 Optimization 문제

여러 modality와 interaction component가 하나의 objective를 중심으로 동시에 학습되면 gradient가 각 component에 서로 다른 영향을 줄 수 있습니다.

```text
                     Loss
                      ↓
              Shared Prediction
                      ↓
                 Interaction
                 ↙         ↘
          Image Encoder   Text Encoder
```

특정 modality가 loss를 빠르게 감소시키면 다른 modality가 충분한 representation을 학습하지 못할 수 있습니다.

또한 각 encoder의 학습 속도, gradient magnitude 및 representation scale이 다르면 optimization imbalance가 발생할 수 있습니다.

따라서 gradient norm, activation distribution, learning dynamics, modality contribution, encoder convergence 및 interaction convergence 등을 별도로 관찰할 필요가 있습니다.

## 16. Fusion Architecture의 구조적 Trade-off

Fusion 방법에 따라 문제의 종류와 정도는 다릅니다.

| 구조 | Identity | Interaction | Observability | 주요 구조적 한계 |
| --- | --- | --- | --- | --- |
| Early Fusion | 낮아질 수 있음 | 초기 단계 | 낮아질 수 있음 | 초기 결합, alignment, noise propagation |
| Feature-level Fusion | 낮아질 수 있음 | 중간 단계 | 낮아질 수 있음 | shared representation, dimension dependency |
| Late Fusion | 높게 유지 가능 | 제한적 | 높게 유지 가능 | fine-grained interaction 부족, model cost |
| Attention 기반 Fusion | 높게 유지 가능 | 강함 | 높게 구성 가능 | interaction complexity, coupling, computational cost |

고도화된 Fusion architecture는 이전 방식의 문제를 완화할 수 있지만 그 과정에서 새로운 구조적 복잡성이 발생할 수 있습니다.

```text
Early Fusion
    ↓
Identity / Alignment 문제

Feature-level Fusion
    ↓
Representation / Coupling 문제

Late Fusion
    ↓
Interaction / Cost 문제

Attention-based Fusion
    ↓
Interaction Graph / Dependency /
Diagnosis / Computational Complexity 문제
```

따라서 하나의 Fusion 방식이 모든 구조적 문제를 동시에 제거한다고 보기 어렵습니다.

## 17. 핵심 문제

Multimodal Fusion에서 중요한 문제는 단순히 feature가 섞이는가에만 있지 않습니다.

보다 근본적으로 다음을 판단할 필요가 있습니다.

1. 각 feature의 identity를 계속 유지할 수 있는가?
2. feature 사이의 interaction을 명시적으로 식별할 수 있는가?
3. 각 component의 상태와 contribution을 독립적으로 관찰할 수 있는가?
4. 새로운 feature를 추가할 때 기존 component의 변경을 최소화할 수 있는가?
5. interaction이 증가해도 구조를 일관되게 확장할 수 있는가?
6. 특정 modality 또는 interaction의 실패를 다른 component와 분리하여 진단할 수 있는가?

따라서 Multimodal Fusion의 구조적 한계는 단순한 information mixing 문제보다 넓게 볼 필요가 있습니다.

핵심은 다음 속성을 동시에 어떻게 유지할 것인가에 있습니다.

**Representation Identity · Component Independence · Explicit Interaction · Observability · Extensibility**

이 문제는 단순 Feature-level Fusion뿐만 아니라 modality identity와 observability를 보존하도록 설계된 Attention 기반 Multimodal Architecture에서도 형태를 달리하여 나타날 수 있습니다.
