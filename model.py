from preprocessing import x_train, x_test, y_train, y_test, test_csv, y_labelEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

param = {'iterations': [1000,1500,2000,2500], 'depth': [4,5,6], 'learning_rate': [0.01,0.05,0.07,0.10]}

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=333)

model = GridSearchCV(CatBoostClassifier(), param, cv=kfold, verbose=1, refit=True, n_jobs=-1)
model.fit(x_train,y_train)

y_pre_best = model.best_estimator_.predict(x_test)
acc = accuracy_score(y_test, y_pre_best)

y_submit = model.best_estimator_.predict(test_csv)
y_submit = y_labelEncoder.inverse_transform(y_submit)

print("최적의 매개변수: ", model.best_estimator_)
print("최적의 파라미터: ", model.best_params_)
print(param)
print("ACC: ",acc)