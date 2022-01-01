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


# Parameters

grid_param = {
'bootstrap': [True],
'max_depth': [80, 90, 100, 110], #max_depth: 樹的最大深度
'max_features': [2, 3], #max_features: 劃分時考慮的最大特徵數，預設auto
'min_samples_leaf': [3, 4, 5], #min_samples_leaf: 分完至少有多少資料才能分
'min_samples_split': [8, 10, 12], #min_samples_split: 至少有多少資料才能再分
'n_estimators': [100, 200, 300, 1000] #n_estimators: 森林中樹木的數量，預設=100
}

# Feature scaler

feature_scaler = StandardScaler()
X_train = feature_scaler.fit_transform(train_X)
X_test = feature_scaler.transform(test_X)

# Random forest 
# Apply 3-fold cross validation and grid search to tune the hyperparameter

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


# Evaluate the model and print the best training score, parameter and testing score of best estimator

acc = accuracy_score(test_y, yhat)

print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, best_result, best_parameters)) 

# reference : https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/ ,
# https://stackabuse.com/cross-validation-and-grid-search-for-model-selection-in-python/

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


