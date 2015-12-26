from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import VotingClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, StandardScaler, PolynomialFeatures
import xgboost as xgb
from sklearn.externals import joblib

import common

def main():
    X_train, X_val, y_train, y_val = common.load_train_dummies()
    print X_train.shape
    clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            colsample_bytree=0.7,
            learning_rate=0.002,
            subsample=0.1,
            seed=484313)
    clf = Pipeline([
        ('vec', MinMaxScaler()),
        ('v2', FeatureUnion([
            ('vec', FunctionTransformer()),
            ('km', KMeans(n_clusters=7)),
            ])
        ),
        ('clf', clf),
    ])
    params = dict(
            clf__n_estimators=[50, 100],
            clf__max_depth=[3, 5],
            clf__learning_rate=[0.02, 0.1],
            clf__subsample=[0.5],
    )
    clf = GridSearchCV(clf, params, scoring='roc_auc', verbose=True)
    common.predict_and_report_val(clf, X_train, X_val, y_train, y_val)
    joblib.dump(clf, 'pickles/xg.pkl')

if __name__ == '__main__':
    main()


