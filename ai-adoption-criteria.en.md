# Criteria for Deciding Whether to Apply AI

Being able to apply AI and **whether applying AI is appropriate** are different questions.

Before deciding to apply AI, at least the following five areas should be checked.

1. Data Availability
2. Evaluation Metric
3. Inference Cost
4. Explainability
5. Maintenance Cost

This document organizes each area around practical questions that can be used in real projects.

---

## 1. Data Availability

**Data Availability** means whether the data required to train and validate an AI model can actually be obtained.

It is not enough to check only whether there is a large amount of data. **Quantity, quality, ground-truth information, representativeness, and usability** should all be examined together.

### Questions to Check

- Does data directly related to the problem exist?
- Can enough data be obtained to train the model?
- For a classification problem, can ground-truth labels be obtained?
- Are incorrect values, missing values, or duplicate data reasonably controlled?
- Does the data include the variety of conditions that occur in the real operating environment?
- Can training, validation, and test data be separated?
- Can the data legally and contractually be used for training with respect to privacy, copyright, and license conditions?
- Can new data continue to be collected after deployment?

### Simple Example

Assume that an AI system for product defect inspection is being developed.

```text
100,000 product images available
        ↓
No defect labels
        ↓
Difficult to use directly for Supervised Learning
```

If the following data is available, applicability can be evaluated more concretely.

```text
Normal product images
Defective product images
Defect-type labels
Various lighting and capture conditions
Sufficient training samples
        ↓
Training / Validation / Test Dataset can be constructed
```

### Core Question

**"Do we have enough reliable data that shows the patterns the AI must learn?"**

If this cannot be answered clearly, the data acquisition strategy should be reviewed before selecting a model architecture.

---

## 2. Evaluation Metric

An **Evaluation Metric** is a quantitative criterion for measuring how well an AI model solves the target problem.

Before applying AI, it must be possible to define **what counts as success**.

### Questions to Check

- Can model success and failure be measured?
- Is the evaluation metric connected to the actual business or operational objective?
- Can a baseline and the AI model be compared under the same conditions?
- Can a minimum performance requirement for operation be defined?
- Are some types of errors more important than others?

### Representative Evaluation Metrics by Problem

| Problem | Possible Evaluation Metrics |
| --- | --- |
| Classification | Accuracy, Precision, Recall, F1 Score |
| Regression | Mean Absolute Error, Mean Squared Error, Root Mean Squared Error |
| Object Detection | Mean Average Precision |
| Ranking / Retrieval | Precision at K, Recall at K, Mean Reciprocal Rank |
| Language Model | Cross-Entropy Loss, Perplexity, task-specific evaluation |

The metric should be selected according to the problem.

For example, in manufacturing inspection where classifying a defective product as normal can cause significant loss, Recall for defective products may matter more than Accuracy alone.

### Core Question

**"Is there an objective measurement that allows us to say the AI has improved?"**

Without a measurable criterion, it is difficult to determine whether the model has actually improved or whether AI adoption is effective.

---

## 3. Inference Cost

**Inference Cost** refers to the time and computing resources required for a trained AI model to receive an input and produce a prediction.

High model performance alone does not guarantee that the model can be deployed in a real service.

### Questions to Check

- Must a prediction be produced within milliseconds or seconds?
- Can it run using only a Central Processing Unit?
- Is a Graphics Processing Unit or another accelerator required?
- How many concurrent requests must be handled?
- Will the model run in the cloud or on an edge device?
- Does the operating device provide enough memory?
- Is the computing cost sustainable as request volume grows?

### Simple Example

```text
Model A
Accuracy: 95%
Inference Time: 30 milliseconds

Model B
Accuracy: 96%
Inference Time: 2 seconds
```

Even though Model B has higher Accuracy, it may be unsuitable for a real-time system that requires responses within 100 milliseconds.

Model performance and inference requirements must therefore be validated together.

### Core Question

**"Can this model actually operate under the required response-time and hardware constraints?"**

---

## 4. Explainability

**Explainability** refers to the degree to which people can understand and analyze why an AI model produced a particular prediction.

Not every AI system requires the same level of explainability. However, as predictions are used in more important decisions, the need to inspect the basis of those predictions may increase.

### Questions to Check

- Is the prediction alone sufficient, or is an explanation also required?
- Must the cause of incorrect predictions be analyzed?
- Must the result be explained to users or operators?
- Is it necessary to analyze which inputs or features influenced a prediction?
- Is it necessary to diagnose model weights, gradients, activations, or attention behavior?

### Simple Example

In a recommendation system, the recommendation result itself may be the main objective.

In manufacturing quality inspection, questions such as the following may be important.

```text
Why was this product classified as defective?
        ↓
Which region or features influenced the decision?
        ↓
Model error?
Data problem?
Capture-condition problem?
```

If explainability is required, methods for analyzing predictions should be designed together with the model architecture rather than considered only after deployment.

### Core Question

**"Does this problem require us to understand not only the AI result, but also why that result was produced?"**

---

## 5. Maintenance Cost

**Maintenance Cost** refers not only to the one-time cost of developing an AI model, but also to the continuing cost of operating and maintaining it.

Unlike conventional software, an AI system may require management of **data, models, preprocessing, evaluation, deployment, and monitoring** in addition to source code.

### Questions to Check

- Must the model be retrained when new data arrives?
- Can the data distribution change over time?
- Can model performance be measured continuously in the operating environment?
- Can model versions and dataset versions be managed?
- Should retraining and validation be automated?
- Are there enough people and resources to operate the model-serving infrastructure?
- If a failure occurs, can the system fall back to an earlier model or rule-based process?

### Simple Example

```text
Conventional Software
Source Code Change
    ↓
Test
    ↓
Deploy
```

An AI system may have more assets and stages to manage.

```text
Data Change
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
Retraining when needed
```

Therefore, AI adoption should not be decided only because the model performs well during development.

### Core Question

**"Can this model continue to be operated and validated after it is developed?"**

---

## 6. Quick Decision Checklist

The following checklist can be used for a quick initial assessment of whether AI is applicable.

| Criterion | Core Question |
| --- | --- |
| Data Availability | Is there enough reliable data for training and validation? |
| Evaluation Metric | Can success and failure be measured objectively? |
| Inference Cost | Can the model operate within the required response-time and hardware constraints? |
| Explainability | Can the basis of predictions be analyzed to the required level? |
| Maintenance Cost | Can the data and model be continuously monitored, validated, and retrained? |

A simplified decision flow is shown below.

```text
Define the problem
        ↓
Is trainable data available?
        │
        ├── No → Acquire Data or Review a Rule-based Approach
        │
        └── Yes
             ↓
Is there an Evaluation Metric?
        │
        ├── No → Define success criteria first
        │
        └── Yes
             ↓
Is inference feasible in the operating environment?
        │
        ├── No → Consider model optimization or another approach
        │
        └── Yes
             ↓
Can the required Explainability be provided?
        │
        ├── No → Review analysis methods or another approach
        │
        └── Yes
             ↓
Is continuous Maintenance feasible?
        │
        ├── No → Reassess operating structure and cost
        │
        └── Yes
             ↓
AI Candidate
             ↓
Compare against a Baseline
             ↓
Decide whether to apply AI
```

The purpose of this checklist is not to evaluate every item as a simple Yes or No. The required level for each criterion depends on the target task and actual operating environment.

Ultimately, AI adoption should be decided only after verifying under the same evaluation conditions that the **AI model provides sufficient practical value compared with an existing software algorithm, rule-based system, or baseline**.
