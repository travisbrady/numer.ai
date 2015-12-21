import json
from time import time
import pandas as pd
from scipy.stats import percentileofscore
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.cross_validation import cross_val_predict

FN = 'numerai_training_data.csv'
FN_TOURN = 'numerai_tournament_data.csv'
X_cols = ['f%d' % (i) for i in range(1, 15)] + ['c1_int']
DB_FN = 'results.json'

def load_train():
    df = pd.read_csv(FN)
    df['c1_int'] = df.c1.apply(lambda x: x.split('_')[1])
    return df

def load_tourn():
    df_tourn = pd.read_csv(FN_TOURN)
    df_tourn['c1_int'] = df.c1.apply(lambda x: x.split('_')[1])
    df_tourn.set_index(['t_id'], inplace=True)
    return df_tourn

def dump_to_db(est, auc, acc, f1, start_time, end_time):
    db = open(DB_FN, 'a')
    out_row = dict(est=est, auc=auc, acc=acc, f1=f1, start_time=start_time, end_time=end_time)
    db.write(json.dumps(out_row) + '\n')
    db.close()

def compare_to_history(auc):
    recs = map(json.loads, open(DB_FN).readlines())
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


if __name__ == '__main__':
    main()

