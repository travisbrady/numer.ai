import xgboost as xgb
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import common

def main():
    df_train = common.load_train()
    X, y = df_train.loc[:, common.X_cols].values, df_train.target.values

    clf = Pipeline([
        ('vec', PolynomialFeatures()),
        ('scale', MinMaxScaler()),
        ('clf', LogisticRegression()),
    ])

    common.predict_and_report(clf, X, y, cv=10)

if __name__ == '__main__':
    main()



