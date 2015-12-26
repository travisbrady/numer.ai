import json
from time import time
import pandas as pd
from scipy.stats import percentileofscore
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.cross_validation import cross_val_predict, cross_val_score

FN = 'numerai_training_data.csv'
FN_TOURN = 'numerai_tournament_data.csv'
X_cols = ['f%d' % (i) for i in range(1, 15)] + ['c1_int']
DB_FN = 'results.json'

def load_train():
    df = pd.read_csv(FN)
    df['c1_int'] = df.c1.apply(lambda x: x.split('_')[1])
    return df

def l():
    df = pd.read_csv(FN)
    dum = pd.get_dummies(df.c1)
    df = pd.concat((df, dum), axis=1)
    df_train, df_val = df[df.validation==0], df[df.validation==1]
    return df_train, df_val

def load_train_dummies():
    df_train, df_val = l()
    X_dum_cols = [c for c in df_train if c[0]=='f' or c.startswith('c1_')]
    X_train = df_train.loc[:, X_dum_cols].values
    X_val = df_val.loc[:, X_dum_cols].values
    y_train, y_val = df_train.target.values, df_val.target.values
    return X_train, X_val, y_train, y_val

def load_tourn_dummies():
    df = pd.read_csv(FN_TOURN)
    dum = pd.get_dummies(df.c1)
    df = pd.concat((df, dum), axis=1)
    df.set_index(['t_id'], inplace=True)
    return df

def load_tourn():
    df_tourn = pd.read_csv(FN_TOURN)
    df_tourn['c1_int'] = df.c1.apply(lambda x: x.split('_')[1])
    df_tourn.set_index(['t_id'], inplace=True)
    return df_tourn

def dump_to_db(est, auc, acc, f1, start_time, end_time, fn=DB_FN):
    db = open(fn, 'a')
    out_row = dict(est=est, auc=auc, acc=acc, f1=f1, start_time=start_time, end_time=end_time)
    db.write(json.dumps(out_row) + '\n')
    db.close()

def compare_to_history(auc, fn=DB_FN):
    recs = map(json.loads, open(fn).readlines())
    df = pd.DataFrame.from_records(recs)
    print '[compare_to_history] N = %d' % (df.shape[0])
    print '[compare_to_history] latest auc of %.6f is in the %.1fth percentile' % (auc, percentileofscore(df.auc.unique(), auc))

def predict_and_report(est, X, y, cv=5):
    t0 = time()
    y_pred = cross_val_predict(est, X, y, cv=cv)
    print confusion_matrix(y, y_pred)
    print classification_report(y, y_pred)
    auc = roc_auc_score(y, y_pred)
    print 'AUC: %f' % (auc)
    t1 = time()
    dump_to_db(repr(est), auc, accuracy_score(y, y_pred), f1_score(y, y_pred), t0, t1)
    compare_to_history(auc)

def predict_and_report_val(est, X_train, X_val, y_train, y_val):
    t0 = time()
    est.fit(X_train, y_train)
    y_pred = est.predict(X_val)
    print confusion_matrix(y_val, y_pred)
    print classification_report(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred)
    print 'AUC: %f' % (auc)
    t1 = time()
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    dump_to_db(repr(est), auc, acc, f1, t0, t1, fn='validation_metrics.json')
    compare_to_history(auc, fn='validation_metrics.json')
