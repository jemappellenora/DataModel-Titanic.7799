import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# set y value
survived = train[train['Survived'] == 1]
not_survived = train[train['Survived'] == 0]
# Gender survival while using age to display survival rate
# ex. sns.displot(total_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color='blue')
total_survived = train[train['Survived']==1]
total_not_survived = train[train['Survived']==0]
male_survived = train[(train['Survived']==1) & (train['Sex']=="male")]
female_survived = train[(train['Survived']==1) & (train['Sex']=="female")]
male_not_survived = train[(train['Survived']==0) & (train['Sex']=="male")]
female_not_survived = train[(train['Survived']==0) & (train['Sex']=="female")]
train_test_data = [train, test]
# Create new columns 'Title' for name ending with .
for dataset in train_test_data:
	dataset['Title']=dataset.Name.str.extract('([A-Za-z]+)\.')
# replace uncommon names
for dataset in train_test_data:
	dataset['Title'] = dataset['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer', 'Dona'], 'Other')
	dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
	dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
	dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
# print(train[['Title', 'Survived']].groupby(['Title'], as_index=True).mean())
# mapping of names
title_mapping = {"Mr":0, "Miss":1, "Mrs":2, "Master":3, "Other":4}
for dataset in train_test_data:
    dataset['Title'] =dataset['Title'].map(title_mapping)
    dataset['Title'] =dataset['Title'].fillna(2)
# mapping for 'Sex'
for dataset in train_test_data:
    dataset['Sex']=dataset['Sex'].map({'female':1,'male':0}).astype(int)
# fill null with highest embarked port 'S'
for dataset in train_test_data: 
    dataset['Embarked']= dataset['Embarked'].fillna('S')
# mapping for embarked
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
# filling null age with mean+-std
for dataset in train_test_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
# Age band calculation
train['AgeBand'] = pd.cut(train['Age'], 5)
# Mapping using Ageband that did split
for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
# fill null fare
for dataset in train_test_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
# split using frequency of Fare
train['FareBand'] = pd.qcut(train['Fare'], 4)
## mapping for Fare
for dataset in train_test_data:
	dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
	dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
	dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
	dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
	dataset['Fare'] = dataset['Fare'].astype(int)
# combine sibsp and parch to make family size
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# we found that lone travelers have 30% survival rate
# here we mark and see probability of survival rate of lone traveler
for dataset in train_test_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize']==1, 'IsAlone'] = 1
# dropping unnecessary features after all mapping is done
features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize']
train = train.drop(features_drop, axis = 1)
test = test.drop(features_drop, axis = 1)
train = train.drop(['PassengerId', 'AgeBand', 'FareBand'], axis=1)

