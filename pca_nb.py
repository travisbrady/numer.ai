from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

import common

def main():
    df_train = common.load_train()
    X, y = df_train.loc[:, common.X_cols].values, df_train.target.values
    clf = make_pipeline(PCA(), GaussianNB())
    clf = BaggingClassifier(base_estimator=clf, max_samples=0.2, n_estimators=25)
    common.predict_and_report(clf, X, y)

if __name__ == '__main__':
    main()


