# Classification Reports — Cancellation Models

## LogisticRegression

- Accuracy: 0.8058
- Precision: 0.7976
- Recall: 0.6375
- F1-score: 0.7086
- ROC-AUC: 0.8743
- Train time: 14.289s

```
               precision    recall  f1-score   support

Not Cancelled       0.81      0.90      0.85     24799
    Cancelled       0.80      0.64      0.71     14591

     accuracy                           0.81     39390
    macro avg       0.80      0.77      0.78     39390
 weighted avg       0.80      0.81      0.80     39390
```

## DecisionTree

- Accuracy: 0.8484
- Precision: 0.8381
- Recall: 0.7321
- F1-score: 0.7815
- ROC-AUC: 0.9265
- Train time: 0.921s

```
               precision    recall  f1-score   support

Not Cancelled       0.85      0.92      0.88     24799
    Cancelled       0.84      0.73      0.78     14591

     accuracy                           0.85     39390
    macro avg       0.85      0.82      0.83     39390
 weighted avg       0.85      0.85      0.85     39390
```

## RandomForest

- Accuracy: 0.8791
- Precision: 0.8912
- Recall: 0.7673
- F1-score: 0.8246
- ROC-AUC: 0.9535
- Train time: 1.956s

```
               precision    recall  f1-score   support

Not Cancelled       0.87      0.94      0.91     24799
    Cancelled       0.89      0.77      0.82     14591

     accuracy                           0.88     39390
    macro avg       0.88      0.86      0.87     39390
 weighted avg       0.88      0.88      0.88     39390
```

## XGBoost

- Accuracy: 0.8820
- Precision: 0.8607
- Recall: 0.8130
- F1-score: 0.8361
- ROC-AUC: 0.9543
- Train time: 1.195s

```
               precision    recall  f1-score   support

Not Cancelled       0.89      0.92      0.91     24799
    Cancelled       0.86      0.81      0.84     14591

     accuracy                           0.88     39390
    macro avg       0.88      0.87      0.87     39390
 weighted avg       0.88      0.88      0.88     39390
```

## MLP

- Accuracy: 0.8631
- Precision: 0.8386
- Recall: 0.7808
- F1-score: 0.8087
- ROC-AUC: 0.9392
- Train time: 18.727s

```
               precision    recall  f1-score   support

Not Cancelled       0.88      0.91      0.89     24799
    Cancelled       0.84      0.78      0.81     14591

     accuracy                           0.86     39390
    macro avg       0.86      0.85      0.85     39390
 weighted avg       0.86      0.86      0.86     39390
```
