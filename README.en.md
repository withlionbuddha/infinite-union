# infinite-union

(Developed while studying train-loop refactoring and multimodal learning in parallel.)

`infinite-union` is a multimodal architecture perspective designed to **preserve the identity and independence of modality-specific features** extracted from different modalities, explicitly construct required interactions, and **extend the system with new modalities or model components**.

Infinite Union is proposed as a multimodal architecture perspective for addressing structural limitations that can arise in multimodal fusion, including degradation of modality-specific feature identity, increased coupling between components, reduced extensibility, and reduced observability.

> **Infinite Union** is not a standard Machine Learning or Deep Learning term. It is a **modeling methodology** proposed in this repository for constructing models in which individual features preserve their independence, connect and interact with one another, and can continue to expand with new features and model components.

---

## 1. Problems Suitable and Unsuitable for AI

Deep Learning is suitable for problems where the relationship between input and output is difficult to describe clearly using fixed rules alone and where representations or patterns must be learned from sufficient data.

Representative examples include:

- Image classification / object detection
- Natural Language Processing
- Speech / audio recognition
- Time-series forecasting / classification
- Anomaly detection
- Multimodal learning

By contrast, for problems where the relationship between input and output can be defined exactly using clear business rules, simple lookup, sorting, or deterministic calculation, a conventional software algorithm or rule-based system may be simpler and easier to verify. AI is not automatically the correct answer for every problem.

Whether to apply AI should be decided based on data availability, evaluation metrics, inference cost, explainability, and maintenance cost.

---

## 2. Major Data Modalities

The primary modalities considered in this repository are shown below.

```text
Image       → Image Encoder       → Image Features
Text        → Text Encoder        → Text Features
Audio       → Audio Encoder       → Audio Features
Time-series → Time-series Encoder → Time-series Features
```

Each encoder learns a representation suitable for its modality.

`Image Features`, `Text Features`, `Audio Features`, and `Time-series Features` are **modality-specific features** produced by different encoders.

---

## 3. Pretrained Model

A pretrained model is first trained on a large-scale dataset and then reused for a downstream task.

Its main advantages are that the cost of learning feature representations from scratch can be reduced and transfer learning can be applied even when the downstream dataset is limited.

Whether a pretrained model should be used must be decided after checking conditions such as the distribution difference between pretraining data and the target domain, model size, inference cost, fine-tuning cost, and license.

In Infinite Union, each modality encoder is treated as an independent component, so pretrained encoders can be selected or replaced separately for each modality.

---

## 4. Multimodal Learning

Multimodal learning is a machine learning approach that uses two or more modalities together to perform a task.

Examples include:

```text
Image + Text
Audio + Text
Image + Time-series
Image + Text + Audio + Time-series
```

One modality can supplement feature representations that are insufficient in another, and relationships between different modalities can be learned.

However, multimodal learning also requires modality-specific preprocessing and encoders. Representation dimensions and temporal or spatial alignment may differ, and missing modalities and additional training or inference costs must be handled.

---

## 5. Decision Criteria for Pretrained Models and Multimodal Learning

Pretrained Models and Multimodal Learning should not be applied automatically. Their use should be decided based on the **target task, dataset, modality, model performance, and compute cost**.

| Criterion | Pretrained Model | Multimodal Learning |
| --- | --- | --- |
| Dataset size | Determine whether training data is limited and whether a pretrained representation appropriate for the target task exists. | Determine whether enough data for each modality and paired/aligned data can be obtained. |
| Reuse of existing models | Check whether a pretrained model applicable to the target task or domain can be selected. | Decide whether modality-specific encoders can be selected or must be trained directly. |
| Input modality | Can be applied to both single and multiple modalities. | Consider use when two or more modality-specific features are required for the target task. |
| Feature necessity | Verify whether reusing a pretrained representation is more effective than task-specific training. | Verify whether different modality-specific features provide additional information required for the task. |
| Domain difference | Check the difference between pretraining and target domains and decide whether fine-tuning or a task-specific model is needed. | Check modality-specific data distributions and alignment conditions and decide how interactions should be applied. |
| Compute cost | Measure training and inference resource requirements and select model size and deployment strategy. | Measure resources required for modality-specific encoders and interaction components. |
| Implementation complexity | Choose among feature extraction, fine-tuning, and full training. | Determine the implementation scope for preprocessing, alignment, missing-modality handling, and interaction mechanisms. |
| Performance validation | Compare against a scratch model or baseline under the same evaluation conditions. | Compare against single-modality baselines and modality ablation results under the same evaluation conditions. |

A simplified decision process is shown below.

```text
Target Task
    │
    ↓
Determine required modalities
    │
    ├── Single Modality
    │       │
    │       ↓
    │   Is an applicable Pretrained Model available?
    │       ├── Yes → Decide whether to use the Pretrained Model
    │       └── No  → Use a Task-specific Model
    │
    └── Multiple Modalities
            │
            ↓
    Are modality-specific features
    required for the target task?
            │
       ┌────┴────┐
       │         │
      No        Yes
       │         │
       ↓         ↓
 Single       Decide whether to use
 Modality     Multimodal Learning
                 │
                 ↓
         Can Pretrained Encoders
         be applied per modality?
                 │
           ┌─────┴─────┐
          Yes          No
           │            │
           ↓            ↓
      Pretrained     Task-specific
      Encoders       Encoders
```

Multimodal Learning should not be selected simply because multiple data types are available. Experiments should verify whether the **modality-specific features are required for the target task** and whether learning interactions among them actually improves model performance over a single-modality baseline.

The need for a multimodal architecture should be evaluated under the same conditions as the single-modality baseline. An **Ablation Study** can add or remove modality-specific features to analyze how each modality and interaction component affects model performance.

---

## 6. Representative Multimodal Architectures

Multimodal architectures can be organized in several ways depending on the task and the point at which interaction occurs. The following categories are representative design examples rather than a fixed taxonomy.

### Early Combination

Low-level inputs or representations are processed together relatively early.

```text
Input A ─┐
         ├→ Shared Processing → Prediction
Input B ─┘
```

### Late Combination

Each modality is processed independently and the results are used together at a downstream stage.

```text
Input A → Model A → Output A ─┐
                              ├→ Decision
Input B → Model B → Output B ─┘
```

### Intermediate Interaction

Relationships between intermediate features produced by modality-specific encoders are computed inside the model.

```text
Input A → Encoder A → Features A ─┐
                                  ├→ Interaction → Prediction
Input B → Encoder B → Features B ─┘
```

---

## 7. Structural Limitations That Can Occur in Multimodal Combination

In a multimodal model, combining several modality representations into one tensor or shared representation can be effective. However, depending on implementation, the following problems may occur.

- It may become difficult to track modality-specific feature identity.
- It may become difficult to replace a particular modality encoder independently.
- Activation or gradient diagnostics for each modality may become more complex.
- Adding a new modality may increase the scope of downstream component changes.
- Missing modalities and modality-specific ablation may become more difficult to handle.

These problems do not necessarily occur in every multimodal architecture and depend on the architecture and implementation.

Infinite Union proposes separating the responsibilities of **feature management and cross-modal interaction** to reduce these design problems.

---

## 8. Infinite Union

The core idea of Infinite Union is not to immediately transform different modality-specific features into a single representation, but to construct a **multimodal architecture that manages them together while preserving the identity of each feature**.

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

`Features Union` is a multimodal architecture concept used in this repository. In Features Union, each feature remains independent and can be identified and accessed separately. In Python, for example, it can be represented using a `dictionary` object.

```python
features_union = {
    "image": image_features,
    "text": text_features,
    "audio": audio_features,
    "time_series": time_series_features,
}
```

The Python `dictionary` object itself is not a new Machine Learning algorithm. The core design constraint is that each modality-specific feature remains **independently identifiable and accessible**.

```text
Features Union
├── Image Features
├── Text Features
├── Audio Features
└── Time-series Features
```

The features are managed together but do not lose their modality identity.

---

## 9. Infinite Extension

Features Union does not fix the architecture concept to a specific number of modalities.

```text
Features Union
├── Image Features
├── Text Features
├── Audio Features
├── Time-series Features
├── New Modality Features
└── ...
```

When a new modality is added, a new encoder and new features are added independently instead of changing existing modality-specific features.

Actual extensibility depends on conditions such as downstream interaction interfaces, dimensional compatibility, training data, and compute resources. `Infinite` does not mean physically unlimited; it represents an **extensible multimodal architecture principle**.

---

## 10. Separation of Features Union and Interaction

The responsibility of Features Union is to **maintain and manage features independently**.

When cross-modal relationship learning is required, a separate `Interaction` component selects and references the required features from the Features Union.

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

Interaction is not fixed to a single algorithm. Depending on the target task and relationships among features, an appropriate mechanism can be selected.

### Attention

Attention computes relevance between Query and Key and uses the result to weight Value.

```text
Attention(Q, K, V) = softmax(QKᵀ / √dₖ)V
```

### Cross-Attention

Cross-Attention computes relationships between different representation sources.

For example, Text Features can be used as Query and Image Features as Key/Value.

```text
CrossAttention(
    Q_text,
    K_image,
    V_image
)
```

### Gating

Gating controls the influence of particular features or modalities.

```text
g = sigmoid(Wx + b)
weighted_features = g × features
```

### Similarity

Similarity measures the similarity between two feature representations.

For example, cosine similarity can be expressed as follows.

```text
similarity(a, b) = (a · b) / (||a|| ||b||)
```

Raw features from different encoders are not automatically comparable. When necessary, projection layers or representation learning can be used to construct a compatible embedding space.

### Feature Selection

Modality-specific features can be selected according to the task or input condition.

```text
Task A → Image Features + Text Features
Task B → Text Features + Time-series Features
Task C → Audio Features
```

Depending on task requirements, selection can be implemented using either deterministic rules or a learned routing mechanism.

---

## 11. Ablation Study

An **Ablation Study** is an experimental method that removes or changes a particular model component or input and compares how model performance changes under the same evaluation conditions.

Because Infinite Union preserves the identity of modality-specific features, experiments that selectively exclude the features of a particular modality can be constructed more easily.

For an Image + Text model, for example, the following conditions can be compared.

```text
1. Image Features + Text Features
2. Image Features only
3. Text Features only
```

From an architecture perspective:

```text
Image Features ─┐
                ├→ Interaction → Prediction
Text Features ──┘
```

The evaluation metric for each modality-specific feature removal condition can be compared with the full condition.

```text
Image + Text
Image only
Text only
```

This makes it possible to **analyze how model performance changes when particular modality-specific features are removed or used**.

However, a performance difference should not automatically be interpreted as the independent contribution of a modality because interaction effects can exist between modality-specific features.

For a more rigorous analysis, the same dataset split, preprocessing, training conditions, and evaluation metrics should be used, and repeated experiments, masking, or component-level ablation can be applied when needed.

---

## 12. Diagnostics / Observability

Preserving the identity of modality-specific features makes modality-level model diagnostics easier to organize.

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

This allows analysis of questions such as:

- Is the gradient of a particular encoder excessively small or large?
- Is the activation distribution of a particular modality abnormal?
- Are the weights of a particular encoder actually being updated?
- Are some modality-specific features rarely used after Interaction?

---

## 13. Architecture Goal

Infinite Union proposes the following structure from a multimodal architecture perspective.

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

The key idea is to **separate feature management from cross-modal relationship computation**.

---

## 14. Design Principles

1. **Independence**  
   Each modality encoder and each modality-specific feature should be manageable independently.

2. **Identity Preservation**  
   It should remain possible to identify the modality from which each feature in the Features Union was generated.

3. **Explicit Interaction**  
   Cross-modal relationships are computed in an explicit Interaction component rather than in Features Union itself.

4. **Extensibility**  
   Changes to existing components should be minimized when adding a new modality or encoder.

5. **Observability**  
   The architecture should support analysis of modality-specific weights, gradients, activations, and interaction behavior.

6. **Replaceability**  
   Interfaces should be separated so that a modality encoder can be replaced with another architecture or pretrained model.

---

## 15. Relationship with deep-learning-core

`deep-learning-core` is responsible for reusable PyTorch training infrastructure, while `infinite-union` defines the multimodal architecture perspective for modality-specific features and cross-modal interactions.

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

The training runner in `deep-learning-core` can be designed to move not only a single Tensor but also nested multimodal input structures such as `dictionary`, `tuple`, and `list` to the device.

Example:

```python
x_batch = {
    "image": image_tensor,
    "text": text_tensor,
    "time_series": time_series_tensor,
}
```

This separates the responsibilities of the `infinite-union` multimodal architecture and reusable training infrastructure.

---

## 16. Vision

The goal of `infinite-union` is not simply to transform several modalities into one representation.

It aims to **design and validate an architecture that preserves the independence and identity of modality-specific features, explicitly constructs the required cross-modal interactions, and can be extended with new modalities and model components**.

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
