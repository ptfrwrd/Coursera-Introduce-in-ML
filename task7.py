from sklearn.svm import SVC
import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )

text = newsgroups.data
number = newsgroups.target
vectorizer = TfidfVectorizer()
data = vectorizer.fit_transform(text)
grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
svc = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(svc, grid, scoring='accuracy', cv=cv)
gs.fit(data, number)
c13 = gs.best_params_
print(c13)

svc_2 = SVC(C = 0.0001, kernel='linear', random_state=241)
gs = GridSearchCV(svc_2, grid, scoring='accuracy', cv=cv)
gs.fit(data, number)

coefs = gs.best_estimator_.coef_
row = abs(coefs.getrow(0).toarray()[0].ravel())
top_ten_indicies = np.argsort(abs(row))[-10:]
top_ten_values = row[top_ten_indicies]
feature_mapping = vectorizer.get_feature_names()
for a in top_ten_indicies:
    print(feature_mapping[a])