from preprocessing import x_train, x_test, y_train, y_test, test_csv, y_labelEncoder
from sklearn.utils import all_estimators
from xgboost import XGBClassifier
import pandas as pd

""" all_algorithms = all_estimators(type_filter='classifier')
# all_algorithms = all_estimators(type_filter='regressor')
# print(len(all_algorithms))  # 41(분류) 55(회귀) 
result_list = []
error_list = []
for name, algorithm in all_algorithms:
    try:
        model = algorithm()
        model.fit(x_train,y_train)
        acc = model.score(x_test,y_test)
    except Exception as e:
        print(f"{name:30} ERROR")
        error_list.append(e)
        continue
    print(f"{name:30} ACC: {acc:.4f}")
    result_list.append((name,acc))
    
# print('error_list: \n',error_list)
best_result = max(result_list)[1]
best_algirithm = result_list[result_list.index(max(result_list))][0]
print(f'\nBest result : {best_algirithm}`s {best_result:.4f}') """

model = XGBClassifier(n_estimators=1000, learning_rate=0.15, max_depth=4, random_state=32)
model.fit(x_train,y_train)

acc = model.score(x_test,y_test)
y_submit = model.predict(test_csv)
y_submit = y_labelEncoder.inverse_transform(y_submit)

submit_csv = pd.read_csv("sample_submission.csv")
print(submit_csv.columns)   # ['id', 'NObeyesdad']
submit_csv['NObeyesdad'] = y_submit

submit_csv.to_csv(f"./submit/acc_{acc:.6f}.csv",index=False)

print("ACC: ",acc)

# n_estimators=1000, learning_rate=0.2, max_depth=4, random_state=32
# ACC:  0.9041425818882466

# n_estimators=1000, learning_rate=0.15, max_depth=4, random_state=32
# ACC:  0.9048651252408478
