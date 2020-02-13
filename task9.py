import pandas as pd
import sklearn.metrics
import numpy as np

data = pd.read_csv('classification.csv')
tp, fp, fn, tn= 0,0,0,0
count = len(data['true'])

for i in range(len(data['true'])):
    if data.iloc[i]['true'] == 1 and data.iloc[i]['pred'] == 1:
        tp += 1
    elif data.iloc[i]['true'] == 1 and  data.iloc[i]['pred'] == 0:
        fn += 1
    elif data.iloc[i]['true'] == 0 and  data.iloc[i]['pred'] == 0:
        tn += 1
    elif data.iloc[i]['true'] == 0 and  data.iloc[i]['pred'] == 1:
        fp += 1
print(tp, fp, fn, tn)

accuracy = sklearn.metrics.accuracy_score(data['true'], data['pred'])
precision = sklearn.metrics.precision_score(data['true'], data['pred'])
recall = sklearn.metrics.recall_score(data['true'], data['pred'])
f_mes = sklearn.metrics.f1_score(data['true'], data['pred'])

print(accuracy, precision, recall, f_mes)

data_scores = pd.read_csv('scores.csv')

score_logreg = sklearn.metrics.roc_auc_score(data_scores['true'], data_scores['score_logreg'])
score_svm = sklearn.metrics.roc_auc_score(data_scores['true'], data_scores['score_svm'])
score_knn = sklearn.metrics.roc_auc_score(data_scores['true'], data_scores['score_knn'])
score_tree = sklearn.metrics.roc_auc_score(data_scores['true'], data_scores['score_tree'])

print(score_logreg, score_svm, score_knn, score_tree)

logreg_precision, logreg_recall, logreg_thresholds =  sklearn.metrics.precision_recall_curve(data_scores['true'], data_scores['score_logreg'])
svm_precision, svm_recall, svm_thresholds =  sklearn.metrics.precision_recall_curve(data_scores['true'], data_scores['score_svm'])
knn_precision, knn_recall, knn_thresholds =  sklearn.metrics.precision_recall_curve(data_scores['true'], data_scores['score_knn'])
tree_precision, tree_recall, tree_thresholds =  sklearn.metrics.precision_recall_curve(data_scores['true'], data_scores['score_tree'])


log = []
svm = []
knn = []
tree =[]
for i in range(len(logreg_recall)):
    if logreg_recall[i] > 0.7:
        log.append(logreg_precision[i])
print('log ', np.max(log))

for i in range(len(svm_recall)):
    if svm_recall[i] > 0.7:
        svm.append(svm_precision[i])
print('svm ', np.max(svm))

for i in range(len(knn_recall)):
    if knn_recall[i] > 0.7:
        knn.append(knn_precision[i])
print('knn ', np.max(knn))

for i in range(len(tree_recall)):
    if tree_recall[i] > 0.7:
        tree.append(tree_precision[i])
print('tree ', np.max(tree))