from sklearn.svm import SVC
import pandas

target_variable = pandas.read_csv('svm-data.csv', header=None).loc[:,0]
features = pandas.read_csv('svm-data.csv', header=None).loc[:, 1:]

clf = SVC(C=100000, kernel='linear', random_state=241)
clf.fit(features,target_variable)
clf.score(features,target_variable)
print(clf.support_)