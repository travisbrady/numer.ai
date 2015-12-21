from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.cross_validation import cross_val_score, cross_val_predict

import common

def main():
    df_train = common.load_train()
    X, y = df_train.loc[:, common.X_cols].values, df_train.target.values
    y_pred = cross_val_predict(GaussianNB(), X, y, cv=5)
    print confusion_matrix(y, y_pred)
    print classification_report(y, y_pred)
    print 'AUC: %f' % (roc_auc_score(y, y_pred))

if __name__ == '__main__':
    main()

