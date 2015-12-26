import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA, RandomizedPCA
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, StandardScaler, PolynomialFeatures
from sklearn.ensemble import BaggingClassifier
from sklearn.externals import joblib

import common

def main():
    X_train, X_val, y_train, y_val = common.load_train_dummies()
    est = LinearDiscriminantAnalysis()
    clf = Pipeline([
        ('vec', PolynomialFeatures(interaction_only=True)),
        ('pca', PCA()),
        ('clf', est)
    ])
    params = dict(
            pca__n_components=[None, 7, 15],
    )
    clf = GridSearchCV(clf, params, scoring='roc_auc', verbose=True, cv=5)
    common.predict_and_report_val(clf, X_train, X_val, y_train, y_val)
    common.predict_and_report_val(clf, X_val, X_train, y_val, y_train)
    print clf.best_params_
    fn = 'pca_lr_val'
    joblib.dump(clf, 'pickles/%s.pkl' % (fn))

if __name__ == '__main__':
    main()





