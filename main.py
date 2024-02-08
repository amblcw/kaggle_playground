from preprocessing import x_train, x_test, y_train, y_test, test_csv, y_labelEncoder
from sklearn.utils import all_estimators
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import pandas as pd

param = {'iterations': 1500, 'depth': 5, 'learning_rate': 0.1}
model = CatBoostClassifier(**param)
model.fit(x_train,y_train)

acc = model.score(x_test,y_test)
y_submit = model.predict(test_csv)
y_submit = y_labelEncoder.inverse_transform(y_submit)

submit_csv = pd.read_csv("sample_submission.csv")
print(submit_csv.columns)   # ['id', 'NObeyesdad']
submit_csv['NObeyesdad'] = y_submit

submit_csv.to_csv(f"./submit/acc_{acc:.6f}.csv",index=False)

print(param)
print("ACC: ",acc)



# n_estimators=1000, learning_rate=0.2, max_depth=4, random_state=32
# ACC:  0.9041425818882466

# n_estimators=1000, learning_rate=0.15, max_depth=4, random_state=32
# ACC:  0.9048651252408478

# {'iterations': 1000, 'depth': 5, 'learning_rate': 0.1}
# ACC:  0.9111271676300579

# {'iterations': 1500, 'depth': 5, 'learning_rate': 0.1}
# ACC:  0.9123314065510597
