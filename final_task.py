import time
import datetime
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


"""
    Градиентный бустинг
"""


features = pd.read_csv('features.csv', index_col='match_id')
features.head()
features.drop(['start_time', 'duration', 'tower_status_radiant', 'tower_status_dire', 'barracks_status_dire',
               'barracks_status_radiant'], axis=1, inplace=True)
matches_count = features.shape[0]
counts = features.count()
missing_values = counts[counts != matches_count]
print("Названия признаков, имеющих пропуски: ")
for pass_data in missing_values.index:
    print(pass_data)
for name in missing_values.index:
    features[name].fillna(0, inplace=True)
target_value = features['radiant_win']
signs_value = features.copy()
features.drop('radiant_win', axis=1, inplace=True)
"""
kf = KFold(n_splits=5, shuffle=True)
scores = defaultdict(list)
for n_estimators in (10, 20, 30):
    clf = GradientBoostingClassifier(n_estimators=n_estimators)
    for train_index, test_index in kf.split(signs_value, target_value):
        start_time = datetime.datetime.now()
        x_train, x_test = signs_value.iloc[train_index], signs_value.iloc[test_index]
        y_train, y_test = target_value.iloc[train_index], target_value.iloc[test_index]

        start_time = datetime.datetime.now()
        clf.fit(x_train, y_train)
        probs = clf.predict_proba(x_test)

        print('Время для n={}: {}'.format(n_estimators, datetime.datetime.now() - start_time))
        scores[n_estimators].append(roc_auc_score(y_test, probs[:, 1]))
    scores[n_estimators] = np.mean(scores[n_estimators])
    print(scores)

"""
"""
    Логистическая регрессия
"""


def score_logistic_regression(signs, target):
    best_score = 0
    best_c = 0
    kf = KFold(n_splits=5, shuffle=True)
    for C in np.logspace(-3, 0, num=5):
        clf = LogisticRegression(penalty='l2', C=C)
        for train_index, test_index in kf.split(signs, target):
            X_train, X_test = signs[train_index], signs[test_index]
            y_train, y_test = target.iloc[train_index], target.iloc[test_index]

            start_time = datetime.datetime.now()

            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)
            print('Время для C={}: {}'.format(C, datetime.datetime.now() - start_time))


            score = roc_auc_score(y_test, probs[:, 1])
            if score > best_score:
                best_score = score
                best_c = C
    return best_score, best_c


scaler = StandardScaler()
x_scaled = scaler.fit_transform(signs_value)
best_score, best_c = score_logistic_regression(x_scaled, target_value)
print("Для регуляризации L-2. Лучший параметр регуляризации (C), наилучшее качество: ")
print(best_score, "\n", best_c, "\n")
signs_value = features.drop(['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero',
                             'd3_hero', 'd4_hero', 'd5_hero'], axis=1)
x_scaled = scaler.fit_transform(signs_value)
best_score, best_c = score_logistic_regression(x_scaled, target_value)
print("Убираем числовые признаки. Лучший параметр регуляризации (C), наилучшее качество: ")
print(best_score, "\n", best_c, "\n")

heroes = ['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero',
          'd3_hero', 'd4_hero', 'd5_hero']
unique_heroes = set()
for hero_feature_name in heroes:
    unique_heroes.update(features[hero_feature_name].unique())
print("сколько различных идентификаторов героев существует в данной игре?")
print(len(unique_heroes), "\n")
unique_heroes = list(unique_heroes)

x_pick = np.zeros((features.shape[0], 108))
for i, match_id in enumerate(features.index):
    for p in range(1, 6):
        r_hero_code = features.loc[match_id, "r%d_hero" % p]
        x_pick[i, unique_heroes.index(r_hero_code)] = 1

        d_hero_code = features.loc[match_id,'d%d_hero' % p]
        x_pick[i, unique_heroes.index(d_hero_code)] = -1
signs_value = np.hstack((signs_value, x_pick))
x_scaled = scaler.fit_transform(signs_value)
best_score, best_c = score_logistic_regression(x_scaled, target_value)
print("После добавления мешка слов:  ")
print(best_score, "\n", best_c, "\n")

print("Постройте предсказания вероятностей победы команды Radiant для тестовой выборки с помощью лучшей из изученных моделей (лучшей с точки зрения AUC-ROC на кросс-валидации).", "\n")
x_test = pd.read_csv('features_test.csv', index_col='match_id')
x_test.drop(['start_time', 'lobby_type' ], axis=1, inplace=True)
matches = x_test.shape[0]
counts = x_test.count()
missing = counts[counts != matches]
for name in missing.index:
    x_test[name].fillna(0, inplace=True)
x_pick = np.zeros((x_test.shape[0], 108))
for i, match_id in enumerate(x_test.index):
    for p in range(1, 6):
        r_hero_code = x_test.loc[match_id, 'r%d_hero' % p]
        x_pick[i, unique_heroes.index(r_hero_code)] = 1

        d_hero_code = x_test.loc[match_id, 'd%d_hero' % p]
        x_pick[i, unique_heroes.index(d_hero_code)] = -1

x_test.drop(['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero',
            'd3_hero', 'd4_hero', 'd5_hero'], axis=1, inplace=True)
x_merged = np.hstack((x_test, x_pick))
x_merged_scaled = scaler.fit_transform(x_merged)
clf = LogisticRegression(penalty='l2', C=1.0)
clf.fit(x_scaled, target_value)
probs = clf.predict_proba(x_merged_scaled)[:, 1]

print("min: ", min(probs),  "\nmax:",  max(probs))
print(probs)
print(probs.shape)
res = {'match_id': [], 'radiant_win': []}
for i, match_id in enumerate(x_test.index):
    res['match_id'].append(match_id)
    res['radiant_win'].append(probs[i])
res_df = pd.DataFrame(res)
res_df.set_index('match_id', inplace=True)
res_df.to_csv('final-results.csv')