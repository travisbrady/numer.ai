import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, StandardScaler, PolynomialFeatures
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import xgboost as xgb

import common

def main():
    X_train, X_val, y_train, y_val = common.load_train_dummies()
    params = dict(
            clf__max_samples=[2000, 6000],
            clf__base_estimator__C=[1.5, 1.2, 1.0],
    )
    est = BaggingClassifier(base_estimator=SVC(), n_estimators=3)
    clf = Pipeline([
        ('vec', StandardScaler()),
        ('pca', PCA()),
        ('clf', est)
    ])
    clf = GridSearchCV(clf, params, scoring='roc_auc', verbose=True, cv=3)
    #common.predict_and_report_val(clf, X_train, X_val, y_train, y_val)
    common.predict_and_report_val(clf, X_val, X_train, y_val, y_train)
    print clf.best_params_
    fn = 'svm_hack_val'
    joblib.dump(clf, 'pickles/%s.pkl' % (fn))

if __name__ == '__main__':
    main()
