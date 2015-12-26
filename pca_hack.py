from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import VotingClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, StandardScaler, PolynomialFeatures
import xgboost as xgb

import common

def main():
    X_train, X_val, y_train, y_val = common.load_train_dummies()
    clf = Pipeline([
        ('pca', PCA()),
        ('clf', GaussianNB())
    ])
    params = dict(
            pca__n_components=[None, 2, 3, 4, 7, 9],
    )
    clf = GridSearchCV(clf, params, scoring='roc_auc', verbose=True)
    common.predict_and_report_val(clf, X_train, X_val, y_train, y_val)

if __name__ == '__main__':
    main()



