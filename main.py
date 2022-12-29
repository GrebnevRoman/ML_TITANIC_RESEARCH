# linear algebra
import numpy as np

# data processing
import pandas as pd

# data visualization
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib import style

# Algorithms
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


def models(X_train, Y_train):
    # Using Logistic Regression Algorithm to the Training Set
    log = LogisticRegression(random_state=0)
    log.fit(X_train, Y_train)

    # Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm

    knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    knn.fit(X_train, Y_train)

    # Using SVC method of svm class to use Support Vector Machine Algorithm

    svc_lin = SVC(kernel='linear', random_state=0)
    svc_lin.fit(X_train, Y_train)

    # Using SVC method of svm class to use Kernel SVM Algorithm

    svc_rbf = SVC(kernel='rbf', random_state=0)
    svc_rbf.fit(X_train, Y_train)

    # Using GaussianNB method of naïve_bayes class to use Naïve Bayes Algorithm
    gauss = GaussianNB()
    gauss.fit(X_train, Y_train)

    # Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm
    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree.fit(X_train, Y_train)

    # Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
    forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    forest.fit(X_train, Y_train)

    # print model accuracy on the training data.
    print('[0]Logistic Regression Training Accuracy:', log.score(X_train, Y_train))
    print('[1]K Nearest Neighbor Training Accuracy:', knn.score(X_train, Y_train))
    print('[2]Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(X_train, Y_train))
    print('[3]Support Vector Machine (RBF Classifier) Training Accuracy:', svc_rbf.score(X_train, Y_train))
    print('[4]Gaussian Naive Bayes Training Accuracy:', gauss.score(X_train, Y_train))
    print('[5]Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))
    print('[6]Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))

    return log, knn, svc_lin, svc_rbf, gauss, tree, forest


pd.set_option('display.max_columns', None)
test_df = pd.read_csv("test.csv")
train_df = pd.read_csv("train.csv")

print(train_df)

# print(train_df.pivot_table('Survived', index='Sex', columns='Pclass'))

train_df.pivot_table('Survived', index='Sex', columns='Pclass').plot()

# Drop the columns
train_df = train_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Remove the rows with missing values
# train_df = train_df.dropna(subset=['embarked', 'age'])

train_df = train_df.dropna(subset=['Embarked', 'Age'])
# print(train_df.isna().sum())

labelencoder = LabelEncoder()

# Encode sex column
train_df.iloc[:, 3] = labelencoder.fit_transform(train_df.iloc[:, 2].values)

# Encode embarked
train_df.iloc[:, 8] = labelencoder.fit_transform(train_df.iloc[:, 7].values)

# Print the NEW unique values in the columns
# Split the data into independent 'X' and dependent 'Y' variables
X = train_df.iloc[:, 2:9].values
Y = train_df.iloc[:, 1].values
# Split the dataset into 80% Training set and 20% Testing set


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# Feature Scaling

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = models(X_train, Y_train)
