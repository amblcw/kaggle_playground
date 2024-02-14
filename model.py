from preprocessing import x_train, x_test, y_train, y_test, test_csv, y_labelEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, RandomizedSearchCV
import optuna

""" param = {'iterations': [900], 'depth': [4], 'learning_rate': [0.06], 'task_type' : ['GPU']}
# param = {'iterations': [500,900, 1500, 2500], 'depth': [4], 'learning_rate': [0.06], 'task_type' : ['GPU']}

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=333)

model = GridSearchCV(CatBoostClassifier(), param, cv=kfold, verbose=1, refit=False, n_jobs=-1)
model.fit(x_train,y_train)

y_pre_best = model.best_estimator_.predict(x_test)
acc = accuracy_score(y_test, y_pre_best)

y_submit = model.best_estimator_.predict(test_csv)
y_submit = y_labelEncoder.inverse_transform(y_submit)

print("최적의 매개변수: ", model.best_estimator_)
print("최적의 파라미터: ", model.best_params_)
print(param)
print("ACC: ",acc) """

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 10.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'thread_count': 4,
        'verbose': False
    }

    model = CatBoostClassifier(**params)

    # Train the model
    model.fit(x_train, y_train, verbose=False)

    # Make predictions on the validation set
    val_preds = model.predict(x_test)

    # Calculate accuracy on the validation set
    accuracy = accuracy_score(y_test, val_preds)

    return accuracy

def objectiveXGB(trial):
    param = {
        'n_estimators' : trial.suggest_int('n_estimators', 500, 4000),
        'max_depth' : trial.suggest_int('max_depth', 8, 16),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 300),
        'gamma' : trial.suggest_int('gamma', 1, 3),
        'learning_rate' : 0.01,
        'colsample_bytree' : trial.suggest_discrete_uniform('colsample_bytree', 0.5, 1, 0.1),
        'nthread' : -1,
        # 'tree_method' : 'gpu_hist',
        # 'predictor' : 'gpu_predictor',
        'lambda' : trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha' : trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'subsample' : trial.suggest_categorical('subsample', [0.6,0.7,0.8,1.0]),
        'random_state' : 1127
    }
    
    # 학습 모델 생성
    model = XGBClassifier(**param)
    xgb_model = model.fit(x_train, y_train, verbose=True) # 학습 진행
    
    # 모델 성능 확인
    score = accuracy_score(xgb_model.predict(x_test), y_test)
    
    return score

# study = optuna.create_study(direction='maximize')
# study.optimize(objectiveXGB, n_trials=100)

# best_params = study.best_params
# print(best_params)

# optuna.visualization.plot_param_importances(study)      # 파라미터 중요도 확인 그래프
# optuna.visualization.plot_optimization_history(study)   # 최적화 과정 시각화

# params = {'iterations': 452, 'learning_rate': 0.18947052287744456, 'depth': 6, 'l2_leaf_reg': 6.8398928223584035, 'border_count': 243} # catboost, always 처리 안함
params = {'iterations': 777, 'learning_rate': 0.10152509183335467, 'depth': 6, 'l2_leaf_reg': 1.4112760375644173, 'border_count': 154} # catboost, always 처리
# params = {'n_estimators': 2391, 'max_depth': 16, 'min_child_weight': 19, 'gamma': 1, 'colsample_bytree': 0.8, 'lambda': 2.7858366632566747, 'alpha': 0.004919261757405025, 'subsample': 0.8}    #xgboost, always 처리
model = CatBoostClassifier(**params)
# model = XGBClassifier(**params)

# Train the model
model.fit(x_train, y_train, verbose=False)

acc = model.score(x_test,y_test)

# Make predictions on the validation set
y_submit = model.predict(test_csv)
y_submit = y_labelEncoder.inverse_transform(y_submit)

# 최적의 매개변수:  <catboost.core.CatBoostClassifier object at 0x000001FEBBDC3EE0>
# 최적의 파라미터:  {'depth': 4, 'iterations': 1000, 'learning_rate': 0.07}
# {'iterations': [1000, 1500, 2000], 'depth': [4, 5, 6], 'learning_rate': [0.01, 0.05, 0.07, 0.1]}
# ACC:  0.9065510597302505