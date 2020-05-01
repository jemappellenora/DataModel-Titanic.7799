from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import itertools
import DataMani

X_train = DataMani.train.drop('Survived', axis=1)
y_train = DataMani.train['Survived']
X_test = DataMani.test.drop('PassengerId', axis=1).copy()

# logistic regression
clf1 = LogisticRegression()
clf1.fit(X_train, y_train)
y_pred_log_reg = clf1.predict(X_test)
acc_log_reg = round( clf1.score(X_train, y_train) * 100, 2)

# support vector machines
clf2 = SVC()
clf2.fit(X_train, y_train)
y_pred_svc = clf2.predict(X_test)
acc_svc = round(clf2.score(X_train, y_train) * 100, 2)

# svc
clf3 = LinearSVC()
clf3.fit(X_train, y_train)
y_pred_linear_svc = clf3.predict(X_test)
acc_linear_svc = round(clf3.score(X_train, y_train) * 100, 2)

# K Neighbors
clf4 = KNeighborsClassifier(n_neighbors = 3)
clf4.fit(X_train, y_train)
y_pred_knn = clf4.predict(X_test)
acc_knn = round(clf4.score(X_train, y_train) * 100, 2)

# Decision Tree
clf5 = DecisionTreeClassifier()
clf5.fit(X_train, y_train)
y_pred_decision_tree = clf5.predict(X_test)
acc_decision_tree = round(clf5.score(X_train, y_train) * 100, 2)

# Ramdom Forest
clf6 = RandomForestClassifier(n_estimators=100)
clf6.fit(X_train, y_train)
y_pred_random_forest = clf6.predict(X_test)
acc_random_forest = round(clf6.score(X_train, y_train) * 100, 2)

# GaussianNB
clf7 = GaussianNB()
clf7.fit(X_train, y_train)
y_pred_gnb = clf7.predict(X_test)
acc_gnb = round(clf7.score(X_train, y_train) * 100, 2)

# Perceptron
clf8 = Perceptron(max_iter=5, tol=None)
clf8.fit(X_train, y_train)
y_pred_perceptron = clf8.predict(X_test)
acc_perceptron = round(clf8.score(X_train, y_train) * 100, 2)

# SGDClassifier
clf9 = SGDClassifier(max_iter=5, tol=None)
clf9.fit(X_train, y_train)
y_pred_sgd = clf9.predict(X_test)
acc_sgd = round(clf9.score(X_train, y_train) * 100, 2)


models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machines', 'Linear SVC', 
              'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes', 
              'Perceptron', 'Stochastic Gradient Decent'],
    
    'Score': [acc_log_reg, acc_svc, acc_linear_svc, 
              acc_knn,  acc_decision_tree, acc_random_forest, acc_gnb, 
              acc_perceptron, acc_sgd]
    })

print(models.sort_values(by='Score', ascending=False))

submission = pd.DataFrame({
        "PassengerId": DataMani.test["PassengerId"],
        "Survived": y_pred_random_forest
    })

submission.to_csv('submission.csv', index=False)

