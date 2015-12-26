import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, RandomizedPCA
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, RandomTreesEmbedding
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC
import common

def fitto(X, y):
    ret = []
    for nc in range(4, 15):
        clf = Pipeline([
            ('vec', FeatureUnion([
                ('pca', PCA(n_components=nc)),
            ])),
            ('clf', LinearSVC()),
            #('clf', GaussianNB())
            #('pca', PCA(n_components=nc)),
        ])
        clf.fit(X, y)
        ret.append(clf)

    return ret

def evalshow(clfs, X, y):
    aucs = []
    for clf, _ in clfs:
        p = clf.predict(X)
        auc = roc_auc_score(y, p[:, 1])
        print auc
        aucs.append(auc)
    return aucs

def predict_tourn(clfs):
    dft = common.load_tourn_dummies()
    X_dum_cols = [c for c in dft if c[0]=='f' or c.startswith('c1_')]
    X_t = dft.loc[:, X_dum_cols]
    preds = [clf.predict(X_t)[:, 1]*wt for clf, wt in clfs]
    pred = np.average(preds, axis=0)
    print pred
    out_df = pd.DataFrame(dict(probability=pred), index=dft.index)
    fn = 'preds_2015_12_23_dumber_lr.csv'
    out_df.to_csv(fn)
    print 'wrote file to ', fn

def main():
    X_train, X_val, y_train, y_val = common.load_train_dummies()
    train_clfs = [(x, 0.7) for x in fitto(X_train, y_train)]
    val_clfs = [(x, 0.3) for x in fitto(X_val, y_val)]
    val_aucs = evalshow(train_clfs, X_val, y_val)
    print
    train_aucs = evalshow(val_clfs, X_train, y_train)
    print
    print 'train_aucs', max(train_aucs)
    print 'val_aucs', max(val_aucs)
    predict_tourn(train_clfs+val_clfs)

if __name__ == '__main__':
    main()

