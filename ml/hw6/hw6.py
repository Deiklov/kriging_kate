from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

np.random.seed(42)

# Импортируйте необходимые классы и функции из соответствующих модулей sklearn


X, y = load_iris(return_X_y=True)

X_train, x_test, y_train, y_test = train_test_split(
    X, y, train_size=0.6, random_state=42)
clf = DecisionTreeClassifier(max_depth=3, random_state=42, criterion='gini')

clf.fit(X_train, y_train)
preds = clf.predict(x_test)

acc = accuracy_score(preds, y_test)

print(acc)
