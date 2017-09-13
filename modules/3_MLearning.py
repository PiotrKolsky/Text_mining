
############## STAGE IV - Models training #################
###########################################################

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection
from sklearn.cross_validation import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from itertools import cycle
from scipy import interp
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from copy import  deepcopy
import os
os.chdir('/home/sas/Zasoby/Python/1Dyplom')


### Data preparation for training purposes
wig = pd.read_csv('Wig.csv')
with open('articles1Days.txt', 'rb') as ap:
    articles = pickle.load(ap)

Xw = [' '.join(a) for a in articles]
Xw.append('pomyślny zwyżka optymizm zysk sukces zarobek wzrost zadowolenie ' * 100) # positive, [1] expected
Xw.append('niepomyślny obniżka pesymizm strata porażka spadek niezadowolenie smutek ' * 100) # negative, [0] expected
Xw.append('brak obojętność realizacja pasywny rozwaga wakacje odpoczynek ' * 100) # neutral, [0] expected

### Vectorisation of words into variables
c = CountVectorizer()
Xa = c.fit_transform(Xw)
Xa.shape

### Data dimensionality reduction
svd = TruncatedSVD(n_components = 6)
Xsvd = svd.fit_transform(Xa)
Xsvd.shape
X = Xsvd[0:len(wig)]; X_test = Xsvd[len(wig):]; y = wig['Gain'][0:]; y_test = [1,0,0]


### Classifiers comparison
### Models preparing
models = []
models.append(('LR', LogisticRegression(C=1)))
models.append(('KNN', KNeighborsClassifier(n_neighbors=5)))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RFC', RandomForestClassifier()))
models.append(('ETC', ExtraTreesClassifier()))
models.append(('AdaBoost', AdaBoostClassifier()))
models.append(('SVMlin', SVC(kernel='linear')))
models.append(('SVMrbf', SVC(kernel='rbf', C=1)))
models.append(('NNlog', MLPClassifier(solver='lbfgs', activation = 'logistic'))) #predicts correctly but very slow
models.append(('NNrelu', MLPClassifier(solver='lbfgs', activation = 'relu'))) #very slow only for dim reducted


### Evaluating each model in a sequence - kfold
seed = 0; scoring = 'accuracy'
results = []; names = []
print('Name  mean_acc  std_acc')
for name, model in models:
    kfold = model_selection.KFold(n_splits=5, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = '{:<8} {:<4.3}  ({:<4.3})'.format(name, cv_results.mean(), cv_results.std())
    print(msg)


### Boxplot algorithm comparison
fig = plt.figure(figsize=(10,6))
fig.suptitle('Algorithm Comparison - kfold')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


### II evaluation - leave one out (very slow!)
results2 = []; names2 = []
scoring = 'accuracy'
loo = LeaveOneOut(len(y))
print('Name  mean_acc  std_acc')
for name, model in models:
    for train_idx, test_idx in loo:
        cv_results = model_selection.cross_val_score(model, X[train_idx], y[train_idx],  scoring=scoring)
        results2.append(cv_results)
        names2.append(name)
    msg = '{:<8} {:<4.3}  ({:<4.3})'.format(name, cv_results.mean(), cv_results.std())
    print(msg)

### Boxplot algorithm comparison
fig = plt.figure(figsize=(10,5))
fig.suptitle('Algorithm Comparison - leave1out')
ax = fig.add_subplot(111)
plt.boxplot(results2)
ax.set_xticklabels(names)
plt.show()

### Forecast using the choosen models
print('Expected [1 0 0]')
for name, model in models:
    msg = '{:<8} {}'.format(name, model.fit(X, y).predict(X_test))
    print(msg)


### GridSearchCV - parameters tuning for choosen classifiers
models2 = []
models2.append(('LR', LogisticRegression(), {'C': [0.1,1,10]}))
models2.append(('CART', DecisionTreeClassifier(), {'max_depth': [3,6,10, None]}))
models2.append(('RFC', RandomForestClassifier(), {'n_estimators':[30,50,70]}))
models2.append(('KNN', KNeighborsClassifier(), {'n_neighbors': [2,3,5]}))
models2.append(('SVC', SVC(probability=True), {'kernel': ['linear','rbf']}))

### Finding the best parameters
bests = []
for name, model, param_grid in models2:
   print('Model: ', name)
   scoring='accuracy'
   grid_search = GridSearchCV(model, param_grid=param_grid, scoring=scoring, n_jobs=2)
   grid_search.fit(X, y)
   res = grid_search.cv_results_
   ### The best params writing down
   bests.append((name, deepcopy(model), grid_search.best_estimator_.get_params()))
   for i in range(len(res['rank_test_score'])):
       print('Rank: %.0f; params: %.20s; mean %s: %.2f; std %s: %.2f'
           %(res['rank_test_score'][i], str(res['params'][i]).strip('{}'),
           scoring, res['mean_test_score'][i],
           scoring, res['std_test_score'][i]))

### The best parameters passing back to the models and prediction
for name, model, param in bests:
    model.set_params(**param)
    model.fit(X, y)
    print(name, 'investment temper prediction, expected [1,0,0], predicted:', model.predict(X_test))


### ROC curves, Area-Under-Curve (AUC) - classifier with cross-validation and plot ROC curves
for name, model, gs in models2:
    cv = StratifiedKFold(n_splits=5)
    mean_tpr = 0.0; mean_fpr = np.linspace(0, 1, 100)

    plt.figure(figsize=(10,5)); lw=2
    colors = cycle(['red', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])

    i = 0
    for (train, test), color in zip(cv.split(X, y), colors):
        probas_ = model.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, color=color,
                 label='ROC fold %d (area = %0.2f)' % (i+1, roc_auc))
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
             label='Random')

    mean_tpr /= cv.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - %s '%name)
    plt.legend(loc="lower right")
    plt.show()


### Final tuning parameters by pipeline
pipeline = Pipeline([
       ('vect', None),
       ('reduce_dim', None ),# TruncatedSVD()),
       ('clf', None)
])

parameters = {
    'vect':([CountVectorizer()]),
    #'vect__max_df': (0.5, 1),
    #'vect__max_features': (100, None), #1000, 10000, 100000, None),
    #'vect__ngram_range': ((1, 1), (1, 2)), #(1,3)),
    'reduce_dim': ([TruncatedSVD()]), #None, TruncatedSVD(10)
    'reduce_dim__n_components': ([6,7,20]),
    'clf': ([LogisticRegression(), SVC(kernel='rbf')]), #  'linear', 'rbf' RandomForestClassifier()  DecisionTreeClassifier()
    #'clf__C': (0.01, 1, 10),
}

### Final learning
grid_search = GridSearchCV(pipeline, parameters, n_jobs=2, verbose=1,
                           scoring='accuracy')
grid_search.fit(Xw[:len(wig)], wig['Gain'])

### The best parameters passed back to the pipeline
print ('The best model results: %0.3f' % grid_search.best_score_)
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print (param_name, best_parameters[param_name])

pipeline.set_params(**best_parameters)
print('Investment temper prediction, expected [1,0,0], predicted:', pipeline.predict(Xw[len(wig):]))

################################################################
