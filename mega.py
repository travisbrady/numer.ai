import numpy as np
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from scipy.stats.mstats import gmean, hmean

import common

def main():
    X_train, X_val, y_train, y_val = common.load_train_dummies()
    slr = make_pipeline(MinMaxScaler(), LogisticRegression())
    plr = make_pipeline(PCA(), LogisticRegression())
    nb_bag = BaggingClassifier(base_estimator=GaussianNB())
    clfs = (
            GaussianNB(),
            #GridSearchCV(slr, dict(logisticregression__C=[1.0, 0.8])),
            make_pipeline(PCA(), GaussianNB()),
            GridSearchCV(plr, dict(pca__n_components=[None, 3, 8], logisticregression__C=[1.0, 0.7]), scoring='roc_auc'),
            GridSearchCV(nb_bag, dict(max_samples=[0.2, 0.4, 0.6], max_features=[0.3, 0.7]), scoring='roc_auc'),
            xgb.XGBClassifier(n_estimators=20, max_depth=3, colsample_bytree=0.7, subsample=0.6, learning_rate=0.1),
            #make_pipeline(KMeans(), GaussianNB()),
            #GridSearchCV(
            #    BaggingClassifier(),
            #    dict(base_estimator=[None, GaussianNB(), LogisticRegression()],
            #        n_estimators=[7, 10, 14],
            #        max_samples=[0.3, 0.6])),
            #GridSearchCV(xgb.XGBClassifier(), dict(n_estimators=[2, 3, 4], learning_rate=[0.01, 0.1], subsample=[0.5, 0.9])),
            #BaggingClassifier(base_estimator=SVC(), max_features=0.8, max_samples=2500, n_estimators=5),
    )
    preds = []
    for clf in clfs:
        print clf
        clf.fit(X_train, y_train)
        val_pred = clf.predict(X_val)
        print roc_auc_score(y_val, val_pred)
        clf.fit(X_val, y_val)
        train_pred = clf.predict(X_train)
        preds.append(np.concatenate((train_pred, val_pred)))
        print roc_auc_score(y_train, train_pred)
        print

    y_all = np.concatenate((y_train, y_val))
    preds = np.column_stack(preds)
    gm = gmean(preds, axis=1)
    hm = hmean(preds+1, axis=1)
    preds = np.column_stack((preds, gm, hm))
    print 'GM', roc_auc_score(y_all, gm)
    print 'HM', roc_auc_score(y_all, hm)
    meta = GaussianNB()
    meta = GridSearchCV(xgb.XGBClassifier(), dict(max_depth=[2, 3, 4], learning_rate=[0.01, 0.05, 0.1], n_estimators=[20, 40, 60]), scoring='roc_auc')
    meta.fit(preds, y_all)
    scores = cross_val_score(meta, preds, y_all, scoring='roc_auc', cv=5)
    print scores
    print scores.mean()

if __name__ == '__main__':
    main()
