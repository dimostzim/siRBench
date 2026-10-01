---
license: other
license_name: tabpfn-3-5-license-v1.0
license_link: LICENSE
extra_gated_fields:
  Organization: text
  Role:
    type: select
    options:
    - Field practitioners
    - Researcher
    - Student
  Use-case: text
  May we contact you about future updates?: checkbox
extra_gated_button_content: Agree to license terms and send request to access repo.
extra_gated_description: "Model weights released under `tabpfn-3-5-license-v1.0`. This license is designed to be permissive for research and internal evaluation. It *explicitly allows* testing, evaluation, and internal benchmarking, so an organization can download the model and run preliminary assessments on its own datasets.\nThe key restriction is that the model, its derivatives, and its outputs cannot be used for any commercial or production purpose. This includes, but is not limited to, revenue-generating products, competitive benchmarking for procurement, client deliverables, or using the model’s results for internal commercial decision-making.\nFor all production use cases, we offer a *Commercial Enterprise License*. This provides access to our proprietary high-speed inference engine, dedicated support, integration tooling, and other internal models. Please contact us at sales@priorlabs.ai for commercial licensing inquiries."
pipeline_tag: tabular-classification
tags:
- chemistry
- biology
- finance
- legal
- climate
- medical
---
### Model Overview
TabPFN-3.5 is a transformer-based foundation model that uses in-context learning to solve tabular prediction problems in a forward pass.
One checkpoint serves both classification and regression.
Inference code can be found at [https://github.com/PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN).

### Getting started
First, install the inference package:
```bash
pip install tabpfn
```

Fitting a classifier and predicting looks like this:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

clf = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5)
clf.fit(X_train, y_train)
prediction_probabilities = clf.predict_proba(X_test)
predictions = clf.predict(X_test)
print("Accuracy", accuracy_score(y_test, predictions))
```

`TabPFNRegressor` works the same way, off the same checkpoint. For more examples, see the github repo: [https://github.com/PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN).

### Checkpoints

| Checkpoint | Description |
| --- | --- |
| [`tabpfn-v3.5-20260909.safetensors`](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/tabpfn-v3.5-20260909.safetensors) | TabPFN-3.5. The default for `ModelVersion.V3_5`. |
| [`tabpfn-v3.5-fast-20260909.safetensors`](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/tabpfn-v3.5-fast-20260909.safetensors) | TabPFN-3.5-Fast, a smaller and faster model. The default for `ModelVersion.V3_5_FAST`. |
| [`tabpfn-v3.5-20260909_multiclass.safetensors`](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/tabpfn-v3.5-20260909_multiclass.safetensors) | TabPFN-3.5 variant. Slightly different architecture, with sometimes better performance on multiclass classification. Experimental. |

To use a specific checkpoint, pass its filename via `model_path`:

```python
from tabpfn import TabPFNClassifier

clf = TabPFNClassifier(model_path="tabpfn-v3.5-20260909_multiclass.safetensors")
```

Every checkpoint ships with basic text and datetime column handling enabled (`TRANSFORM_TEXT`, `TRANSFORM_DATES`) and a feature limit of 20,000.

### Developers & Affiliations
Developed by Prior Labs.

### Intended Use
Regression and classification tasks on structured tabular data, including string and datetime columns.

### Not Intended Use
- Not suitable for unstructured data such as images.

### Training Data and Priors
TabPFN-3.5 is trained purely on synthetic tabular tasks.

### Ethical Considerations
Having been trained purely on synthetic datasets, TabPFN-3.5 is free from dataset leakage from the pretraining stage.
However, like for any other tabular prediction method, when applied to high-risk use cases, users should ensure that the labelled data is free of biases.

### Licensing
Model weights released under tabpfn-3-5-license-v1.0.

The license is designed to be permissive for research and limited internal evaluation. It *explicitly allows* testing, evaluation, and internal benchmarking, so an organization can download the model and run preliminary assessments on its own datasets.
The key restriction is that the model, its derivatives, and its outputs cannot be used for any commercial or production purpose. This includes, but is not limited to, revenue-generating products, competitive benchmarking for procurement, client deliverables, or using the model’s results for internal commercial decision-making.
For all production use cases, we offer a *Commercial Enterprise License*. This provides access to our proprietary high-speed inference engine, dedicated support, integration tooling, and other internal models.
Please contact us at sales@priorlabs.ai for commercial licensing inquiries.

### Version
v1.0: initial release.
