#!/usr/bin/env python3
# -*- coding:utf8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

logistic = linear_model.LogisticRegression()
pca = decomposition.PCA()
pipeline = Pipeline(steps=[("pca", pca), ("logistic", logistic)])
print(pipeline)

digits = datasets.load_digits()
X = digits.data
Y = digits.target

# pca plot
pca.fit(X)
plt.figure(figsize=(4, 3))
plt.axes([.2, .2, .7, .7])
plt.plot(np.cumsum(pca.explained_variance_ratio_), linewidth=2)
plt.hlines(0.9, 0.0, pca.n_components_, colors='r')
plt.axis('tight')
plt.xlabel('#num of components')
plt.ylabel('explained variances')
plt.yticks(np.arange(0.0, 1.1, 0.1))

# grid search
n_components = [20, 40, 64]
Cs = np.logspace(-4, 4, 3)
print("n_components:", n_components)
print("Cs:", Cs)

estimator = GridSearchCV(pipeline,
                        {"pca__n_components": n_components,
                         "logistic__C": Cs})
estimator.fit(X, Y)
best_model = estimator.best_estimator_
plt.axvline(best_model.named_steps["pca"].n_components_,
            linestyle=":", label="n_components chosen")
# show figure
plt.show()
