from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble.bagging import BaggingClassifier

Models = {
    'LogisticRegression': LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                                             intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                                             penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                                             verbose=0, warm_start=False),
    'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=9, random_state=123,
                                                     splitter="best", criterion="gini"),
    'KNeighborsClassifier': KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                                                 metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                                                 weights='uniform'),
    'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=123,
                                                     max_depth=9, criterion="gini"),

    'GaussianNB': GaussianNB(priors=None),
    'SVC': SVC(C=1.0, kernel='linear', probability=True, random_state=124),
    'MLPClassifier': MLPClassifier(alpha=1, max_iter=1000, random_state=124),
    'BaggingClassifier': BaggingClassifier(random_state=124)

}
