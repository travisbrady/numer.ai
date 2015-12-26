import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.grid_search import GridSearchCV
import common

def main():
    X_train, X_val, y_train, y_val = common.load_train_dummies()
    clf = make_pipeline(KMeans(), GaussianNB())
    clf = Pipeline([
        ('km', KMeans()),
        ('clf', xgb.XGBClassifier())
    ])
    params = dict(
            km__n_clusters=[7, 10, 20],
            clf__n_estimators=[15, 30]
    )
    clf = GridSearchCV(clf, params, verbose=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    print roc_auc_score(y_val, y_pred)
    print clf.best_params_

if __name__ == '__main__':
    main()
