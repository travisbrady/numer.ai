import pandas as pd
import xgboost as xgb
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, FunctionTransformer
from sklearn.pipeline import Pipeline, make_union, FeatureUnion
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.grid_search import GridSearchCV

FN = 'numerai_training_data.csv'
FN_TOURN = 'numerai_tournament_data.csv'
X_cols = ['f%d' % (i) for i in range(1, 15)] + ['c1_int']

def load_data():
    df = pd.read_csv(FN)
    df['c1_int'] = df.c1.apply(lambda x: x.split('_')[1])

    df_train, df_test = df[df.validation==0], df[df.validation==1]
    X_train, X_test = df_train.loc[:, X_cols], df_test.loc[:, X_cols]
    y_train, y_test = df_train.target, df_test.target
    df_tourn = pd.read_csv(FN_TOURN)
    df_tourn['c1_int'] = df.c1.apply(lambda x: x.split('_')[1])
    df_tourn.set_index(['t_id'], inplace=True)
    return X_train.values, X_test.values, y_train.values, y_test.values, df_tourn

def main():
    X_train, X_test, y_train, y_test, df_tourn = load_data()
    clf = BaggingClassifier(base_estimator=LogisticRegression(), n_estimators=1000, max_features=0.5, max_samples=0.2)
    clf = LogisticRegression()
    clf = RandomForestClassifier()
    clf = BaggingClassifier(RandomForestClassifier(), n_estimators=250, max_features=0.6, max_samples=0.6)
    clf = GaussianNB()
    clf = BaggingClassifier(base_estimator=GaussianNB(), n_estimators=2000, max_features=0.3, max_samples=0.3)

    clf = Pipeline([
        ('vec', make_union(
            FunctionTransformer(),
            PCA(n_components=3),
            KMeans(),
        )),
        ('scale', StandardScaler()),
        ('clf', xgb.XGBClassifier(n_estimators=800, learning_rate=0.05, subsample=0.7)),
    ])

    params = dict(max_depth=[5, 7],
            learning_rate=[0.02, 0.1, 0.2],
            n_estimators=[100, 300, 500],)
    #clf = GridSearchCV(xgb.XGBClassifier(), params)
    voter = VotingClassifier(estimators=[
        ('lr', LogisticRegression()),
        ('nb', GaussianNB()),
        ('rf', RandomForestClassifier()),
        ('bag', BaggingClassifier(base_estimator=GaussianNB(), max_features=0.5, max_samples=0.25)),
        ('svm', BaggingClassifier(base_estimator=SVC(), max_samples=0.1)),
        ('hub', BaggingClassifier(base_estimator=SGDClassifier(loss='modified_huber', n_iter=20), max_samples=0.5)),
        ('xgb', xgb.XGBClassifier(n_estimators=20, subsample=0.5)),
    ], voting='hard')

#    clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.2, max_depth=9, subsample=0.8)
#    clf.fit(X_train, y_train, eval_metric='auc', eval_set=[(X_test, y_test)], early_stopping_rounds=10)

    params = dict(
            max_features=[0.5, 1.0],
            max_samples=[0.5, 1.0],
            n_estimators=[10, 30,],
    )
    #clf = GridSearchCV(BaggingClassifier(base_estimator=GaussianNB()), params, cv=5)
    est = BaggingClassifier(base_estimator=LogisticRegression())
    est = BaggingClassifier(base_estimator=voter)
    clf = GridSearchCV(est, params)
    print clf
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print 'AUC', roc_auc_score(y_test, y_pred)
    y_tourn = clf.predict_proba(df_tourn.loc[:, X_cols].values)

    y_tourn_df = pd.DataFrame(y_tourn[:, 1], columns=['probability'], index=df_tourn.index)
    y_tourn_df.to_csv('numerai_predictions.csv')


if __name__ == '__main__':
    main()

