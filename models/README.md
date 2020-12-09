# Evaluate Models
## Dec 09, 2020
## Author RIHAD

### Compute and interpret ROC curve(s) and AUC score(s)
ROC (Receiver Operating Characteristic) curve(s) is used to evaluate different thresholds for classification ML tasks. ROC curve visualizes a confusion matrix for every threshold

### What are thresholds?
Each time we train a classification model, we access *prediction probabilities*. If a probability > 0.5, the instance is classified as positive. Here, 0.5 is the decision threshold. You can adjust it to reduce the number of false positives or false negatives

ROC curve shows a false positive rate on the X-axis. This metric informs us about the proportion of negative class classified as positive

On the Y-axis, it shows a true positive rate. This metric is sometimes called Recall or Sensitivity. It informs us about the positive class proportion that was correctly classified

Refer to the following images: confusion matrix and TPR/FPR calculation:

<img src="./asset/img1.png"/>

### What is AUC?
AUC represents the area under the ROC curve. The higher the AUC, the better the model is at correctly classifying instances. Ideally, the ROC curve should extend to the top left corner. The AUC score would be 1 in that scenario

<img src="./asset/auc.png"/>
