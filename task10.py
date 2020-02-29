from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge

'''work with data'''
data = pd.read_csv('salary-train.csv')
data_test = pd.read_csv('salary-test-mini.csv')
target_train = data['SalaryNormalized']
target_test = data_test['SalaryNormalized']


''' to lower case'''
data['FullDescription'] = data['FullDescription'].str.lower()
data_test['FullDescription'] = data_test['FullDescription'].str.lower()


'''replace everything expect letters and numbers with spaces'''
data['FullDescription'] = data['FullDescription'].str.replace('[^a-zA-Z0-9]', ' ', regex = True)
data_test['FullDescription']  = data_test['FullDescription'] .str.replace('[^a-zA-Z0-9]', ' ', regex = True)


'''replace pass with "nan" '''
data['LocationNormalized'] = data['LocationNormalized'].fillna('nan')
data['ContractTime']  = data['ContractTime'].fillna('nan')
data_test['LocationNormalized'] = data_test['LocationNormalized'].fillna('nan')
data_test['ContractTime']  = data_test['ContractTime'].fillna('nan')

'''TF-IDF min_df = 5'''
vectorizer = TfidfVectorizer(min_df=5)
full_description = vectorizer.fit_transform(data['FullDescription'])
test_full_description = vectorizer.transform(data_test['FullDescription'])


''' DictVectorizer for one-hot-coding'''
enc = DictVectorizer()
X_train_categ = enc.fit_transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

'''using hstack'''
X_train = hstack((full_description, X_train_categ))
X_test = hstack((test_full_description, X_test_categ))


'''ridge regression'''
clf = Ridge(alpha=1.0, random_state=241)
clf.fit(X_train, target_train)
print(clf.predict(X_test))