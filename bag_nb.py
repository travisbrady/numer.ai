from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.grid_search import GridSearchCV

import common

def main():
    df_train = common.load_train()
    X, y = df_train.loc[:, common.X_cols].values, df_train.target.values
    est = BaggingClassifier(base_estimator=GaussianNB())
    params = dict(
            max_features=[0.4, 0.6, 0.8],
            max_samples=[0.4, 0.6, 0.8],
            n_estimators=[8, 18],
            bootstrap=[False, True]
    )
    clf = GridSearchCV(est, params, scoring='roc_auc')
    clf.fit(X, y)
    common.predict_and_report(clf, X, y)

if __name__ == '__main__':
    main()


