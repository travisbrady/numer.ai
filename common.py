import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.cross_validation import cross_val_predict

FN = 'numerai_training_data.csv'
FN_TOURN = 'numerai_tournament_data.csv'
X_cols = ['f%d' % (i) for i in range(1, 15)] + ['c1_int']

def load_train():
    df = pd.read_csv(FN)
    df['c1_int'] = df.c1.apply(lambda x: x.split('_')[1])
    return df

def load_tourn():
    df_tourn = pd.read_csv(FN_TOURN)
    df_tourn['c1_int'] = df.c1.apply(lambda x: x.split('_')[1])
    df_tourn.set_index(['t_id'], inplace=True)
    return df_tourn

def predict_and_report(est, X, y, cv=5):
    y_pred = cross_val_predict(est, X, y, cv=cv)
    print confusion_matrix(y, y_pred)
    print classification_report(y, y_pred)
    print 'AUC: %f' % (roc_auc_score(y, y_pred))

if __name__ == '__main__':
    main()

