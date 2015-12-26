import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier, RandomTreesEmbedding
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_predict, train_test_split, KFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import common

class StackingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.base_estimators = [
                GaussianNB(),
                BaggingClassifier(base_estimator=GaussianNB(), max_features=0.3, max_samples=0.1, n_estimators=17),
                LogisticRegression(C=0.8),
                xgb.XGBClassifier(n_estimators=30, max_depth=3, subsample=0.2, learning_rate=0.01),
        ]
        self.meta_estimator = LogisticRegression()
        self.meta_estimator = GaussianNB()

    def fit(self, X, y):
        base_preds = []
        for be in self.base_estimators:
            be_pred = []
            for train_index, test_index in KFold(X.shape[0], n_folds=5, random_state=4150):
                be.fit(X[train_index], y[train_index])
                #be_fold_pred = be.predict_proba(X[test_index])
                be_fold_pred = be.predict(X[test_index])
                be_pred.append(be_fold_pred)
            be_pred = np.concatenate(be_pred)
            #be_pred = cross_val_predict(be, X, y, cv=5)
            base_preds.append(be_pred)
            be.fit(X, y)
        base_preds = np.column_stack(base_preds)
        self.meta_estimator.fit(base_preds, y)

    def predict_proba(self, X):
        bases = []
        for be in self.base_estimators:
            bases.append(be.predict(X))
        bases = np.column_stack(bases)
        p = self.meta_estimator.predict_proba(bases)
        return p

    def predict(self, X):
        bases = []
        for be in self.base_estimators:
            print be
            #bases.append(be.predict(X))
            #bases.append(be.predict_proba(X))
            bases.append(be.predict(X))
        bases = np.column_stack(bases)
        p = self.meta_estimator.predict(bases)
        return p

def main():
    X_train, X_val, y_train, y_val = common.load_train_dummies()
    clf = Pipeline([
        ('pca', PCA()),
        ('clf', StackingClassifier())
    ])
    clf.fit(X_train, y_train)
    fn = 'stack'
    joblib.dump(clf, 'pickles/%s.pkl' % (fn))
    y_pred = clf.predict(X_val)
    print y_val[:10]
    print y_pred[:10]
    print roc_auc_score(y_val, y_pred)
    hack = joblib.load('pickles/pca_hack_2.pkl')
    h2 = joblib.load('pickles/rpca_pca_hack_2.pkl')
    y_hack = hack.predict(X_val)
    y_h2 = h2.predict(X_val)
    duh = (y_pred + y_hack + y_h2) / 3.0
    print 'duh auc', roc_auc_score(y_val, duh)


if __name__ == '__main__':
    main()

