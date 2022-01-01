# 11010COM525200 Financial Technology Program HW3

## 必要的套件 Needed Python Extension Packages (Python 3.9.7)

must import **yfinanc , numpy , pandas , talib ,  matplotlib , sklearn** Packages in order to run the program

For **HW3_P1.py** you should import :
```python
import yfinance as yf
import numpy as np
import pandas as pd
import talib as ta
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
```

For **HW3_P2.py** you should import :
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn import ensemble
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle
```

## 程式說明 Program Description

本次作業的程式檔案：**HW3_P1.py** (For Problem 1)、**HW3_P2.py**（For Problem 2）安裝了必要的套件（如上）直接運行即可，分別完成了Problem 1以及Problem 2所要求的功能，具體註解已經標註在程式上。**(PS:必須先運行HW_P1.py以獲得TAIEX.csv數據)**

**HW3_P1.py** (For Problem 1)
  
**Problem 1A**
```python
 # Problem 1A
    # Collect the TAIEX from 2012/12/01 to 2021/12/02 (Day Bar)
    
    data = yf.download('%5ETWII', start='2012-12-01', end='2021-12-02')
    data.drop('Adj Close', axis='columns', inplace=True)
    data = data[order]
    data.columns = order_l
```
**Problem 1B**
```python
 # Problem 1B
    # Set labels and (apply stop loss / profit taking) pt_sl
    
    labels = [1,2,0]
    PTSL = [0.04, 0.02] 
    min_ret = 0.0005

    close = data['close']

    # Adding vertical barriers in 20 days
    
    vertical_barriers = add_vertical_barrier(t_events=close.index , close=close, num_days=20) 
    
    # Implement triple barrier method and label

    triple_barrier_events = get_events(close=close,
                                  t_events=close.index,
                                  pt_sl=PTSL,
                                  target=close,
                                  min_ret=min_ret,
                                  vertical_barrier_times=vertical_barriers,
                                  label = ['1', '2', '0'])
    
    data = data.assign(label = triple_barrier_events['label'])
```
**Problem 1C**
```python
# Problem 1C
    # (i) Bias of moving average for 5-days, 10-days, 20-days, 60-day
    
    SMA5 = ta.SMA(close, timeperiod = 5)
    SMA10 = ta.SMA(close, timeperiod = 10)
    SMA20 = ta.SMA(close, timeperiod = 20)
    SMA60 = ta.SMA(close, timeperiod = 60)
    data['bias5'] = (close - SMA5)/SMA5
    data['bias10'] = (close - SMA10)/SMA10
    data['bias20'] = (close - SMA20)/SMA20
    data['bias60'] = (close - SMA60)/SMA60

    # (ii) RSI : 14
    data['RSI14'] = ta.RSI(close, timeperiod = 14)

    # (iii) MACD, MACD signal, MACD histogram
    data['MACD'], data['MACD_signal'], data['MACD_hist'] = \
        ta.MACDFIX(close, signalperiod=9)
    
    # (iv) Save problem 1,2,3 results to csv
    data.to_csv('./TAIEX.csv')
    
```

**HW3_P2.py** (For Problem 2)

**Splitting data and get the distribution of data**
```python
# Load file

file = './TAIEX.csv'

df1 = pd.read_csv(file, header = 0 ,usecols=[0,2,3,4,5], index_col=[0])
df2 = pd.read_csv(file, header = 0 ,usecols=[0,6], index_col=[0])

na1 = df1.to_numpy()
na2 = df2.to_numpy()

# Splitting training data and testing data

data = na1
target = na2 

target = label_binarize(target, classes=[0,1,2])
n_classes = 3

train_X, test_X, train_y, test_y = train_test_split(data,target,test_size = 0.3)

# The distribution of all data, training data and testing data

print(train_X.shape)
print(test_X.shape)
print(train_y.shape)
print(test_y.shape)
```
**Implement of given parameter**
```python
# Parameters

grid_param = {
'bootstrap': [True],
'max_depth': [80, 90, 100, 110], #max_depth: 樹的最大深度
'max_features': [2, 3], #max_features: 劃分時考慮的最大特徵數，預設auto
'min_samples_leaf': [3, 4, 5], #min_samples_leaf: 分完至少有多少資料才能分
'min_samples_split': [8, 10, 12], #min_samples_split: 至少有多少資料才能再分
'n_estimators': [100, 200, 300, 1000] #n_estimators: 森林中樹木的數量，預設=100
}
```
**Feature scaler**
```python
feature_scaler = StandardScaler()
X_train = feature_scaler.fit_transform(train_X)
X_test = feature_scaler.transform(test_X)
```

**Random forest Methon and Apply 3-fold cross validation and grid search to tune the hyperparameter**
```python
forest = ensemble.RandomForestClassifier(random_state=1)

cv = KFold(n_splits=3, shuffle=True, random_state=1)

gd_sr = GridSearchCV(estimator=forest,
                     param_grid=grid_param,
                     scoring='accuracy',
                     cv=cv,
                     n_jobs=-1)
                    
gd_sr.fit(X_train,train_y)

best_parameters = gd_sr.best_params_
best_result = gd_sr.best_score_
best_model = gd_sr.best_estimator_
yhat = best_model.predict(X_test)
```
**Evaluate the model and print the best parameter**
```python
acc = accuracy_score(test_y, yhat)
print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, best_result, best_parameters)) 
```

**Plot of a ROC curve for a specific class** （Graph：ROC_Curve_Figure.png）
```python
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
lw=2

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_y[:, i], yhat[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class

colors = cycle(['blue', 'red', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()

```

以上為程式說明，具體的結果報告在 Financial Technology HW3 Assignment.pdf 上

## 參考 Reference
https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/ \
https://stackabuse.com/cross-validation-and-grid-search-for-model-selection-in-python/ \
*Methon reference from : Book by Marcos López de Prado 'Advances in Financial Machine Learning'*
