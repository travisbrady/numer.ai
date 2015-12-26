import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA, RandomizedPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, StandardScaler, PolynomialFeatures
from sklearn.ensemble import BaggingClassifier
from sklearn.externals import joblib
import xgboost as xgb

import common

def main():
    X_train, X_val, y_train, y_val = common.load_train_dummies()
    params = dict(
            n_estimators=[50, 100, 150],
            max_depth=[None, 3, 5, 6],
    )
    clf = RandomForestClassifier()
    clf = GridSearchCV(clf, params, scoring='roc_auc', verbose=True, cv=3)
    #common.predict_and_report_val(clf, X_train, X_val, y_train, y_val)
    common.predict_and_report_val(clf, X_val, X_train, y_val, y_train)
    print clf.best_params_
    fn = 'rf2_val'
    joblib.dump(clf, 'pickles/%s.pkl' % (fn))

if __name__ == '__main__':
    main()





