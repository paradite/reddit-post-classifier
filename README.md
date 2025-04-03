# Reddit Post Classifier

This project is a simple classifier for Reddit posts. It uses a pre-trained model to classify posts as relevant or irrelevant.

Created by [16x Tracker](https://tracker.16x.engineer/)

## Results

### polar-wave-5

- Trained for 3 epochs

5746015ef3a8bc2661ef1db0e135539847d113b5

```
               precision    recall  f1-score   support

           0       0.97      0.99      0.98      1437
           1       0.50      0.26      0.34        50

    accuracy                           0.97      1487
   macro avg       0.74      0.63      0.66      1487
weighted avg       0.96      0.97      0.96      1487
```

### ruby-vortex-6

- Trained for 10 epochs
- WeightedRandomSampler

b894522593fdae0422b3c0c2a4ac9c8dd30f2bf3

```
               precision    recall  f1-score   support

           0       0.98      0.95      0.96      1437
           1       0.25      0.50      0.33        50

    accuracy                           0.93      1487
   macro avg       0.62      0.72      0.65      1487
weighted avg       0.96      0.93      0.94      1487
```