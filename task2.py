import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas

data = pandas . read_csv('train.csv')
workData = pandas.DataFrame({"Pclass": data['Pclass'] , "Fare": data['Fare'], "Age": data['Age'], "Sex": data['Sex'], "Survived": data['Survived']})
workData['Sex'] = np.where(workData['Sex'] == 'male',0,1)
workData = workData.dropna(subset=['Age'])
clf = DecisionTreeClassifier(random_state=241)
clf.fit(workData[['Pclass','Fare','Age','Sex']], workData['Survived'])
for a,b in zip(workData[['Pclass','Sex','Age','Fare']].columns, clf.feature_importances_):
    print(a,b)

