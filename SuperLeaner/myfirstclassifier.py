# Add more packages as required
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import tree, metrics, ensemble
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV, KFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.datasets import load_iris
from scipy.stats import mode
# Create a new classifier which is based on the sckit-learn BaseEstimator and ClassifierMixin classes

models = {}
test_index_list = []
kfold = KFold(5)
k = 5
class Classifier(BaseEstimator, ClassifierMixin):
    """An ensemble classifier that uses heterogeneous models at the base layer and
        a aggregatnio model at the aggregation layer. A k-fold cross validation is
        used to gnerate training data for the stack layer model.
    """
    option1 = ["Decision Tree", "Random Forest", "SVM", "KNN", "Navie Bayes", "Logistic Regression"]
    # Constructor for the classifier object
    def __init__(self,training_type = True, layer_type = "Decision Tree", estimators = option1):
        """Setup a SuperLearner classifier
        Parameters
        training_type - choose the label based data or probability based data to train stack layer model
        layer_type - choose a models for stack layer(Decision Tree, Random Forest, SVM, Navie Bayes or KNN)
        estimators - choose 5-10 base classifier model
        """
        self.superlearner = None
        self.training_type = training_type
        self.layer_type = layer_type
        self.estimators = estimators
        self.final_training_data = None
        self.labels = None
        self.baselearners = {"Decision Tree" : DecisionTreeClassifier(), "Random Forest": RandomForestClassifier(),
                 "SVM" :SVC(probability=True), "KNN": KNeighborsClassifier(), "Navie Bayes": GaussianNB(),
                 "GDBT": GradientBoostingClassifier(), "Logistic Regression": LogisticRegression()}
        # print(self.baselearners)
    # The fit function to train a classifier

    def fit_baselearner(self, X, y):
        # use K-fold method to train the base classifier
        training_data = []
        true_label = []
        transformed_data = {}
        for key in self.estimators:
            models[key] = []
            transformed_data[key] = []

        for k, (train_index, test_index) in enumerate(kfold.split(X, y)):
            true_label.append(y[test_index])  # append to label of test data to list
            test_index_list.append(test_index)
            k_number = []
            # k_store.append(k)
            for key in self.estimators:
                # print(key)
                if key in self.baselearners:
                    model = clone(self.baselearners[key])
                    # print(model.__hash__())
                    model = model.fit(X[train_index], y[train_index])
                    models[key].append(model)

                if self.training_type == True:
                    pred = model.predict(X[test_index])
                    transformed_data[key].append(pred)

                if self.training_type == False:
                    proba = model.predict_proba(X[test_index])
                    transformed_data[key].append(proba)
            # print(transformed_data)

        # print(len(models))
        # construct data
        label = np.concatenate(true_label)
        for key1 in transformed_data:
            transformed_data[key1] = np.concatenate(transformed_data[key1])
            training_data.append(transformed_data[key1])
        if self.training_type == True:
            training_data_handling = np.array(training_data).T
            final_training_data = np.c_[training_data_handling, label]
        if self.training_type == False:
            training_data_handling = training_data[0]
            for i in range(1,len(training_data)):
                training_data_handling = np.concatenate([training_data_handling,training_data[i]], axis = 1)
                final_training_data = np.c_[training_data_handling, label]
        # print(final_training_data.shape[0])
        return final_training_data

    def fit(self, X, y):
        training_data = self.fit_baselearner(X, y)
        # print(training_data.shape)
        # print(training_data)
        if self.layer_type == "Decision Tree":
            self.layer_type = self.baselearners["Decision Tree"]
        self.superlearner = self.layer_type.fit(training_data[:,:(training_data.shape[1]-1)], training_data[:,training_data.shape[1]-1])
        return self

    def generate_test_data(self, X):
        if self.training_type == True:
            testing_data = {}
            alist = []
            for key in models:
                testing_data[key] = []
                for model in models[key]:
                    testing_data[key].append(model.predict(X))
                    data_handling = np.array(testing_data[key]).T
                majority_voting = mode(data_handling, axis=-1)[0]
                alist.append(majority_voting)
                final_testing_data = np.concatenate(alist, axis = 1)
            print(final_testing_data.shape)
            return final_testing_data

        if self.training_type  == False:
            testing_data = {}
            alist = []
            for key in models:
                testing_data[key] = []
                for model in models[key]:
                    testing_data[key].append(model.predict_proba(X))
            for key in testing_data:
                data_handling = np.array(testing_data[key])
                data_ = data_handling[0]
                for i in range(1,len(data_handling)):
                    data_ = data_ +data_handling[i]
                final_ = data_/ k
                alist.append(final_)
            testing_data_handling = alist[0]
            for a in range(1, len(alist)):
                testing_data_handling = np.concatenate([testing_data_handling, alist[a]], axis=1)
            print(testing_data_handling.shape)
            return testing_data_handling

    def predict(self, X):
        # print((self.generate_test_data(X)).shape)
        pred = self.superlearner.predict(self.generate_test_data(X))
        # print(pred)
        return pred

clf = Classifier(training_type = False)
iris = load_iris()
# test = clf.fit_baselearner(iris.data, iris.target)
# test.predict(iris.data)
test = clf.fit(iris.data, iris.target)
pred = test.predict(iris.data)
score = metrics.accuracy_score(pred,iris.target)
print(score)
# x = clf.predict(iris.data)
# clf.test_stack_layer(iris.data)
# clf.test_stack_layer(iris.data)
# clf.test_stack_layer(iris.data)
# score  = cross_val_score(clf, X = iris.data, y = iris.target, cv = 5)
# print(score)
#
# clf1 = SuperLearnerClassifier(training_type = False)
# score1 = cross_val_score(clf1, X = iris.data, y = iris.target, cv = 5)
# print(score1)