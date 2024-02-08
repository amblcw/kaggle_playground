from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
import pandas as pd
import numpy as np

train_csv = pd.read_csv("train.csv", index_col=0)
test_csv = pd.read_csv("test.csv", index_col=0)
submit_csv = pd.read_csv("sample_submission.csv")

# print(train_csv.shape, test_csv.shape, submit_csv.shape)    # (20758, 17) (13840, 16) (13840, 2)
# print(train_csv.head)
'''
<bound method NDFrame.head of        
        Gender      Age    Height      Weight   family_history_with_overweight FAVC      FCVC     NCP        CAEC   SMOKE    CH2O   SCC    FAF       TUE       CALC                 MTRANS           NObeyesdad
id
0        Male  24.443011  1.699998   81.669950                            yes  yes  2.000000  2.983297   Sometimes    no  2.763573  no  0.000000  0.976473  Sometimes  Public_Transportation  Overweight_Level_II
1      Female  18.000000  1.560000   57.000000                            yes  yes  2.000000  3.000000  Frequently    no  2.000000  no  1.000000  1.000000         no             Automobile        Normal_Weight
2      Female  18.000000  1.711460   50.165754                            yes  yes  1.880534  1.411685   Sometimes    no  1.910378  no  0.866045  1.673584         no  Public_Transportation  Insufficient_Weight
3      Female  20.952737  1.710730  131.274851                            yes  yes  3.000000  3.000000   Sometimes    no  1.674061  no  1.467863  0.780199  Sometimes  Public_Transportation     Obesity_Type_III
4        Male  31.641081  1.914186   93.798055                            yes  yes  2.679664  1.971472   Sometimes    no  1.979848  no  1.967973  0.931721  Sometimes  Public_Transportation  Overweight_Level_II
...       ...        ...       ...         ...                            ...  ...       ...       ...         ...   ...       ...  ..       ...       ...        ...                    ...                  ...
20753    Male  25.137087  1.766626  114.187096                            yes  yes  2.919584  3.000000   Sometimes    no  2.151809  no  1.330519  0.196680  Sometimes  Public_Transportation      Obesity_Type_II
20754    Male  18.000000  1.710000   50.000000                             no  yes  3.000000  4.000000  Frequently    no  1.000000  no  2.000000  1.000000  Sometimes  Public_Transportation  Insufficient_Weight
20755    Male  20.101026  1.819557  105.580491                            yes  yes  2.407817  3.000000   Sometimes    no  2.000000  no  1.158040  1.198439         no  Public_Transportation      Obesity_Type_II
20756    Male  33.852953  1.700000   83.520113                            yes  yes  2.671238  1.971472   Sometimes    no  2.144838  no  0.000000  0.973834         no             Automobile  Overweight_Level_II
20757    Male  26.680376  1.816547  118.134898                            yes  yes  3.000000  3.000000   Sometimes    no  2.003563  no  0.684487  0.713823  Sometimes  Public_Transportation      Obesity_Type_II
'''
print(train_csv.columns)
# ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
#        'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
#        'CALC', 'MTRANS', 'NObeyesdad']
x_labelEncoder = LabelEncoder()
train_csv['Gender'] = x_labelEncoder.fit_transform(train_csv['Gender'])
train_csv['family_history_with_overweight'] = x_labelEncoder.fit_transform(train_csv['family_history_with_overweight'])
train_csv['FAVC'] = x_labelEncoder.fit_transform(train_csv['FAVC'])
train_csv['CAEC'] = x_labelEncoder.fit_transform(train_csv['CAEC'])
train_csv['SMOKE'] = x_labelEncoder.fit_transform(train_csv['SMOKE'])
train_csv['SCC'] = x_labelEncoder.fit_transform(train_csv['SCC'])
train_csv['CALC'] = x_labelEncoder.fit_transform(train_csv['CALC'])
train_csv['MTRANS'] = x_labelEncoder.fit_transform(train_csv['MTRANS'])

y_labelEncoder = LabelEncoder()
train_csv['NObeyesdad'] = y_labelEncoder.fit_transform(train_csv['NObeyesdad'])
# print(train_csv.head)
'''
<bound method NDFrame.head of        
       Gender     Age      Height     Weight   family_history_with_overweight    FAVC    CVC       NCP     CAEC  SMOKE   CH2O    SCC    FAF       TUE      CALC   MTRANS    NObeyesdad
id
0           1  24.443011  1.699998   81.669950                               1     1  2.000000  2.983297     2      0  2.763573    0  0.000000  0.976473     1       3           6
1           0  18.000000  1.560000   57.000000                               1     1  2.000000  3.000000     1      0  2.000000    0  1.000000  1.000000     2       0           1
2           0  18.000000  1.711460   50.165754                               1     1  1.880534  1.411685     2      0  1.910378    0  0.866045  1.673584     2       3           0
3           0  20.952737  1.710730  131.274851                               1     1  3.000000  3.000000     2      0  1.674061    0  1.467863  0.780199     1       3           4
4           1  31.641081  1.914186   93.798055                               1     1  2.679664  1.971472     2      0  1.979848    0  1.967973  0.931721     1       3           6
...       ...        ...       ...         ...                             ...   ...       ...       ...   ...    ...       ...  ...       ...       ...   ...     ...         ...
20753       1  25.137087  1.766626  114.187096                               1     1  2.919584  3.000000     2      0  2.151809    0  1.330519  0.196680     1       3           3
20754       1  18.000000  1.710000   50.000000                               0     1  3.000000  4.000000     1      0  1.000000    0  2.000000  1.000000     1       3           0
20755       1  20.101026  1.819557  105.580491                               1     1  2.407817  3.000000     2      0  2.000000    0  1.158040  1.198439     2       3           3
20756       1  33.852953  1.700000   83.520113                               1     1  2.671238  1.971472     2      0  2.144838    0  0.000000  0.973834     2       0           6
20757       1  26.680376  1.816547  118.134898                               1     1  3.000000  3.000000     2      0  2.003563    0  0.684487  0.713823     1       3           3
'''