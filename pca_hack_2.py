import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import VotingClassifier, BaggingClassifier, ExtraTreesClassifier, RandomTreesEmbedding
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, RandomizedPCA
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, StandardScaler, PolynomialFeatures
from sklearn.externals import joblib
import xgboost as xgb

import common

def main():
    X_train, X_val, y_train, y_val = common.load_train_dummies()
    est = BaggingClassifier(base_estimator=GaussianNB())
    clf = Pipeline([
        ('pca', RandomizedPCA()),
        ('clf', est)
    ])
    params = dict(
            pca__n_components=[None, 4, 7, 9],
            pca__whiten=[True, False],
            clf__max_samples=[0.9],
            clf__max_features=[0.5, 0.9, 1.0],
            clf__bootstrap=[False],
            clf__n_estimators=[10, 15, 25],
    )
    clf = GridSearchCV(clf, params, scoring='roc_auc', verbose=True, cv=5)
    #common.predict_and_report_val(clf, X_train, X_val, y_train, y_val)
    common.predict_and_report_val(clf, X_val, X_train, y_val, y_train)
    print clf.best_params_
    fn = 'rpca_pca_hack_2_val'
    joblib.dump(clf, 'pickles/%s.pkl' % (fn))

if __name__ == '__main__':
    main()



