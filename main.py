from preprocessing import x_train, x_test, y_train, y_test, test_csv, y_labelEncoder
from sklearn.utils import all_estimators
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import pandas as pd
from model import y_submit, acc
import datetime

# param = {'iterations': 1000, 'depth': 4, 'learning_rate': 0.07}
# model = CatBoostClassifier(**param)
# model.fit(x_train,y_train)

# acc = model.score(x_test,y_test)
# y_submit = model.predict(test_csv)
# y_submit = y_labelEncoder.inverse_transform(y_submit)

submit_csv = pd.read_csv("sample_submission.csv")
print(submit_csv.columns)   # ['id', 'NObeyesdad']
submit_csv['NObeyesdad'] = y_submit

dt = datetime.datetime.now()
submit_csv.to_csv(f"./submit/{dt.day}acc_{acc:.6f}.csv",index=False)

# print(param)
print("ACC: ",acc)



# n_estimators=1000, learning_rate=0.2, max_depth=4, random_state=32
# ACC:  0.9041425818882466

# n_estimators=1000, learning_rate=0.15, max_depth=4, random_state=32
# ACC:  0.9048651252408478

# {'iterations': 1000, 'depth': 5, 'learning_rate': 0.1}
# ACC:  0.9111271676300579

# {'iterations': 1500, 'depth': 5, 'learning_rate': 0.1}
# ACC:  0.9123314065510597

# 열 제거 안함
# 최적의 파라미터:  {'depth': 4, 'iterations': 900, 'learning_rate': 0.06, 'task_type': 'GPU'}
# {'iterations': [900], 'depth': [4], 'learning_rate': [0.06], 'task_type': ['GPU']}
# ACC:  0.9132947976878613
# Index(['id', 'NObeyesdad'], dtype='object')
# ACC:  0.9132947976878613
# 실제 ACC: 0.8945

# FAVC만 제거
# 최적의 파라미터:  {'depth': 4, 'iterations': 900, 'learning_rate': 0.06, 'task_type': 'GPU'}
# {'iterations': [900], 'depth': [4], 'learning_rate': [0.06], 'task_type': ['GPU']}
# ACC:  0.9096820809248555
# Index(['id', 'NObeyesdad'], dtype='object')
# ACC:  0.9096820809248555
# 실제 ACC: 0.444

# SMOKE만 제거
# 최적의 파라미터:  {'depth': 4, 'iterations': 900, 'learning_rate': 0.06, 'task_type': 'GPU'}
# {'iterations': [900], 'depth': [4], 'learning_rate': [0.06], 'task_type': ['GPU']}
# ACC:  0.9135356454720617
# Index(['id', 'NObeyesdad'], dtype='object')
# ACC:  0.9135356454720617

# CatBoostClassifier, SCC 제거
# ACC:  0.9146558293116587