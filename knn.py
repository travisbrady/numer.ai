import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import xgboost as xgb

import common

def main():
    X_train, X_val, y_train, y_val = common.load_train_dummies()
    params = dict(
            clf__n_neighbors=[1, 3, 5, 7],
            clf__weights=['distance', 'uniform'],
    )
    clf = Pipeline([
        ('vec', MinMaxScaler()),
        ('clf', KNeighborsClassifier())
    ])
    clf = GridSearchCV(clf, params, scoring='roc_auc', verbose=True, cv=3)
    #common.predict_and_report_val(clf, X_train, X_val, y_train, y_val)
    common.predict_and_report_val(clf, X_val, X_train, y_val, y_train)
    print clf.best_params_
    fn = 'knn_val'
    joblib.dump(clf, 'pickles/%s.pkl' % (fn))

if __name__ == '__main__':
    main()






