# Structural Limitations and Problems of Multimodal Fusion

Multimodal Fusion is a common approach that combines information generated from different modalities such as image, text, audio, and time-series data so that it can be used for a single prediction or decision.

This does not mean that Fusion itself is an incorrect approach. Various structures exist, including Early Fusion, Feature-level Fusion, Late Fusion, and Attention-based Fusion, and each has different strengths and limitations.

However, when multiple modalities or features are organized around a Fusion structure, structural limitations may arise depending on the Fusion method.

## 1. Limitations of Early Fusion

Early Fusion combines raw inputs or early-stage features from different modalities at a relatively early stage.

```text
Image ───────┐
Text ────────┼──→ Early Fusion → Shared Model → Prediction
TimeSeries ──┘
```

This structure can be simple, but because modality-specific information is combined from an early stage, it may become difficult to independently trace the influence of each modality in later representations.

In addition, modalities with different input structures, sampling rates, sequence lengths, scales, or dimensionalities may require substantial alignment and preprocessing before they can be combined.

Major limitations include:

- Possible loss of modality-specific identity
- Need for heterogeneous input alignment
- Increased modality-specific preprocessing dependencies
- Possibility that noise from one modality affects processing of other modalities from an early stage
- Difficulty in independently diagnosing each modality
- Potential need to modify input and early network structures when adding a new modality

## 2. Limitations of Feature-level Fusion

Feature-level Fusion combines features produced by individual encoders and then creates a shared representation.

```text
Image Encoder ──→ image_features ──────┐
                                       │
Text Encoder ───→ text_features ───────┼─→ Concat/Add → Fusion Layer
                                       │                  ↓
Audio Encoder ──→ audio_features ──────┘           fused_features
```

For example, concatenation can be used.

```python
fused_features = torch.cat(
    [image_features, text_features, audio_features],
    dim=-1,
)

output = fusion_layer(fused_features)
```

Immediately after `torch.cat()`, each modality can still be distinguished because the location of each feature segment is known. However, after passing through a Linear Layer, Multilayer Perceptron, or another transformation, a shared representation is produced.

```text
image_features ─┐
text_features ──┼─→ Fusion → [f1, f2, f3, ..., fn]
audio_features ─┘
```

At this point, it may become difficult to directly determine whether a specific latent feature originated from one modality or from a combination of multiple modalities.

Major limitations include:

- Possible loss of modality-specific feature identity
- Difficulty tracing feature provenance
- Possible reduction in observability of individual modality representations
- Component coupling to the fusion layer
- Structural dependency on the fusion input dimension
- Potential need to modify the fusion layer when adding a new modality
- Architecture compatibility problems with existing checkpoints

For example, combining 256-dimensional Image features and 256-dimensional Text features produces a 512-dimensional Fusion input. Adding 128-dimensional Time-series features increases the Fusion input to 640 dimensions.

```python
# Before
nn.Linear(512, 512)

# After adding Time-series
nn.Linear(640, 512)
```

Therefore, adding a new modality can propagate structural changes into existing Fusion components.

## 3. Limitations of Late Fusion

Late Fusion processes each modality relatively independently and combines results at the prediction, probability, score, or decision level.

```text
Image Model ──→ Prediction ──┐
                             │
Text Model ───→ Prediction ──┼─→ Late Fusion → Final Prediction
                             │
Audio Model ──→ Prediction ──┘
```

This approach can preserve modality-specific model independence and observability relatively well. However, it may not learn sufficiently fine-grained low-level or intermediate-level interactions between modalities.

Major limitations include:

- Limited fine-grained cross-modal interaction
- Limited sharing of intermediate representations across modalities
- Cost of maintaining separate models
- Need for weighting or calibration when combining multiple predictions
- Possible bias caused by differences in confidence scales across modalities
- Increased model and inference cost as the number of modalities grows

Late Fusion therefore reduces some identity-related problems, but introduces other trade-offs in interaction capability and computational cost.

## 4. Limitations of Attention-based Fusion

Attention- or Cross-Attention-based architectures can preserve modality-specific representations while explicitly computing relationships between modalities.

```text
image_features ─────────────┐
                            ├─→ Cross-Attention
text_features ──────────────┘
                            ↓
                  image_text_features
```

This can reduce feature identity and observability problems that may occur in simple Feature-level Fusion. However, preserving the original representations does not remove all structural problems.

### 4.1 Growth of Interaction Components

With only Image and Text, one pairwise interaction can be defined. When Audio and Time-series modalities are added, the number of possible relationships increases.

```text
Image ↔ Text
Image ↔ Audio
Image ↔ TimeSeries
Text  ↔ Audio
Text  ↔ TimeSeries
Audio ↔ TimeSeries
```

If every pairwise interaction is explicitly constructed, the number of unordered pairs for `M` modalities is:

```text
M(M - 1) / 2
```

If directional Cross-Attention relationships are implemented separately, up to `M(M - 1)` directed interaction relationships may need to be considered.

Therefore, interaction graph complexity can grow as the number of modalities increases.

### 4.2 Interaction Identity Complexity

Attention-based structures can preserve the identity of the original features.

```text
image_features
text_features
audio_features
```

However, if interaction outputs are also managed independently, more representations may continue to appear.

```text
image_text_features
image_audio_features
text_audio_features
image_timeseries_features
text_timeseries_features
audio_timeseries_features
...
```

As a result, while the identity problem of the original modalities is reduced, a new problem emerges: managing the identity and lifecycle of interaction representations.

## 5. Movement of Component Coupling

Using Attention does not eliminate component dependency.

```python
cross_attention(
    query=text_features,
    key=image_features,
    value=image_features,
)
```

This interaction can depend on interfaces such as feature dimension, token structure, embedding dimension, projection dimension, mask structure, sequence length, and normalization.

For example, if an Image Encoder output changes as follows:

```text
[B, 196, 768]
        ↓
[B, 256, 1024]
```

projection layers, adapters, or attention interfaces may need to be changed.

In other words, coupling may not disappear; it can move from Fusion Layer coupling to Interaction Interface coupling.

## 6. Observability and Diagnosis Complexity

Attention-based architectures can expose image features, text features, Query/Key/Value projections, attention scores, attention weights, attention outputs, residual outputs, and other states for inspection.

This is an advantage in terms of observability. However, more observable information does not necessarily mean that diagnosis becomes simpler.

When performance degrades, many states may need to be checked.

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

Therefore, the following two ideas are not equivalent.

```text
Higher Observability ≠ Lower Diagnosis Complexity
```

Observability can improve while the number of components and states that must be diagnosed also increases.

## 7. Modality Dominance

The presence of all modalities in the architecture does not guarantee that every modality contributes meaningfully to prediction.

```text
Modality Presence ≠ Guaranteed Modality Contribution
```

Training may cause one modality to dominate prediction while other modalities contribute very little.

To verify this, additional analysis may be required, including modality ablation, gradient analysis, activation analysis, attention distribution analysis, representation similarity analysis, and comparison of modality-specific prediction contributions.

Preserving modality identity and guaranteeing modality contribution are therefore separate problems.

## 8. Missing Modality Problem

In real environments, every modality may not always be available.

```text
Training:  Image + Text + Audio
Inference: Image + Text + Missing Audio
```

If a Fusion architecture strongly depends on the simultaneous presence of several modalities, performance may change substantially when a modality is missing.

Separate strategies may therefore be required, such as masking, imputation, fallback paths, modality dropout, or missing-modality training.

## 9. Modality Quality Imbalance

The quality of each modality is not always equal.

```text
Image       → high quality
Text        → high quality
Audio       → noisy
TimeSeries  → partially missing
```

Because Fusion combines these signals, a low-quality modality can influence the overall representation or prediction.

Therefore, it may be necessary to evaluate not only whether a modality exists, but also its quality, reliability, confidence, noise, and missingness.

## 10. Representation Alignment Problem

Different modalities inherently have different representation meanings and structures.

```text
Image       [B, patches, embedding]
Text        [B, tokens, embedding]
Audio       [B, frames, embedding]
TimeSeries  [B, timesteps, features]
```

Projecting them to the same dimension does not mean that they are semantically aligned.

```text
Same Dimension ≠ Same Semantics
```

Representation alignment itself can therefore become a separate learning problem in multimodal Fusion.

## 11. Temporal and Spatial Alignment Problems

When video, audio, sensor, and time-series data are used together, the same event may be observed at different temporal or spatial resolutions.

```text
Video frame     t = 10.2 sec
Audio segment   t = 10.0 ~ 10.5 sec
Sensor sample   t = 10.17 sec
Text event      timestamp unclear
```

Before Fusion or during interaction, the model must determine which pieces of information correspond to one another. Incorrect alignment can cause incorrect cross-modal relationships to be learned.

## 12. Increased Computational Cost

Attention-based multimodal architectures require not only encoder computation for each modality, but also projection and interaction computation.

```text
Encoder Cost
     +
Projection Cost
     +
Interaction Cost
     +
Fusion / Prediction Cost
```

Token- or sequence-level Cross-Attention in particular can require substantial memory and computation as sequence length grows.

Applying separate interactions to multiple modality pairs can further increase parameters, memory usage, training time, inference latency, and interaction computation.

## 13. Extensibility Problems

Assume that a new Sensor modality is added to an existing Image, Text, and Audio architecture.

In a Fusion-centered architecture, several of the following parts may need to change.

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

Therefore, being able to add a modality is different from being able to extend the architecture without changing existing components.

```text
Can Add a Modality ≠ Structurally Extensible
```

When evaluating extensibility, the scope of changes required in existing components matters in addition to whether a new component can be added.

## 14. Ablation and Root-Cause Analysis Complexity

To determine which component caused a performance change in a multimodal model, many combinations may need to be compared.

```text
Image
Text
Audio
Image + Text
Image + Audio
Text + Audio
Image + Text + Audio
```

When interaction components are also included, the number of experimental combinations can increase further.

```text
Image + Text without interaction
Image + Text with Cross-Attention
Image + Audio with Cross-Attention
...
```

As modalities and interactions increase, the cost of Ablation Studies required to verify the cause of model performance changes can also increase.

## 15. Training Dependencies and Optimization Problems

When multiple modalities and interaction components are trained simultaneously around a shared objective, gradients may affect each component differently.

```text
                     Loss
                      ↓
              Shared Prediction
                      ↓
                 Interaction
                 ↙         ↘
          Image Encoder   Text Encoder
```

If one modality reduces the loss much faster, other modalities may fail to learn sufficiently useful representations.

Differences in learning speed, gradient magnitude, and representation scale across encoders can also create optimization imbalance.

It may therefore be necessary to separately inspect gradient norms, activation distributions, learning dynamics, modality contributions, encoder convergence, and interaction convergence.

## 16. Structural Trade-offs of Fusion Architectures

The type and degree of limitations differ depending on the Fusion method.

| Architecture | Identity | Interaction | Observability | Main Structural Limitations |
| --- | --- | --- | --- | --- |
| Early Fusion | May decrease | Early stage | May decrease | Early combination, alignment, noise propagation |
| Feature-level Fusion | May decrease | Intermediate stage | May decrease | Shared representation, dimension dependency |
| Late Fusion | Can remain high | Limited | Can remain high | Lack of fine-grained interaction, model cost |
| Attention-based Fusion | Can remain high | Strong | Can be high | Interaction complexity, coupling, computational cost |

More advanced Fusion architectures can reduce problems found in earlier approaches, but may introduce new forms of structural complexity.

```text
Early Fusion
    ↓
Identity / Alignment problems

Feature-level Fusion
    ↓
Representation / Coupling problems

Late Fusion
    ↓
Interaction / Cost problems

Attention-based Fusion
    ↓
Interaction Graph / Dependency /
Diagnosis / Computational Complexity problems
```

No single Fusion approach should therefore be assumed to eliminate all structural problems at once.

## 17. Core Problems

The important issue in Multimodal Fusion is not simply whether features are mixed.

More fundamentally, the following questions need to be addressed.

1. Can the identity of each feature continue to be preserved?
2. Can interactions between features be explicitly identified?
3. Can the state and contribution of each component be observed independently?
4. Can new features be added while minimizing changes to existing components?
5. Can the structure continue to expand consistently as interactions increase?
6. Can a failure in a specific modality or interaction be diagnosed separately from other components?

The structural limitations of Multimodal Fusion therefore need to be considered more broadly than as a simple information-mixing problem.

The central challenge is how to preserve the following properties simultaneously:

**Representation Identity · Component Independence · Explicit Interaction · Observability · Extensibility**

These problems can appear not only in simple Feature-level Fusion, but also in different forms within Attention-based Multimodal Architectures designed to preserve modality identity and observability.
