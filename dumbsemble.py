import glob
import numpy as np
import pandas as pd
import xgboost as xgb
from stack import StackingClassifier
from sklearn.externals import joblib
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score
import common

def avg_pred(clfs, X):
    preds = []
    for clf in clfs:
        yp = clf.predict_proba(X)
        preds.append(yp)
    y_pred = np.average(preds, axis=0)
    return y_pred

def main():
    preds, clfs = [], []
    X_train, X_val, y_train, y_val = common.load_train_dummies()
    for fn in glob.glob('pickles/*.pkl'):
        print fn
        clf = joblib.load(fn)
        y_pred = clf.predict(X_val)
        preds.append(y_pred)
        clfs.append(clf)

    avgd_pred = np.average(preds, axis=0)
    meta = xgb.XGBClassifier(n_estimators=90, subsample=0.6, colsample_bytree=0.5, learning_rate=0.05)
    predsa = np.column_stack(preds)
    Xx = np.column_stack((X_val, predsa))
    meta_pred = cross_val_predict(meta, Xx, y_val, cv=3)
    meta_auc = roc_auc_score(y_val, meta_pred)
    avg_auc = roc_auc_score(y_val, avgd_pred)
    common.compare_to_history(avg_auc, fn='validation_metrics.json')
    common.compare_to_history(meta_auc.mean(), fn='validation_metrics.json')
    meta.fit(Xx, y_val)
    yolo = (avgd_pred + meta_pred) / 2.0
    print 'yolo', roc_auc_score(y_val, yolo)

    dft = common.load_tourn_dummies()
    X_dum_cols = [c for c in dft if c[0]=='f' or c.startswith('c1_')]
    X_t = dft.loc[:, X_dum_cols]
    y_pred_t = avg_pred(clfs, X_t)
    X_m_t = np.column_stack((X_t, y_pred_t))
    y_pred_m = meta.predict_proba(X_m_t)
    out_df = pd.DataFrame(dict(probability=y_pred_m[:, 1]), index=dft.index)
    out_df.to_csv('preds_2015_12_23.csv')


if __name__ == '__main__':
    main()
