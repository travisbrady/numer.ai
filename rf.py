from sklearn.ensemble import RandomForestClassifier

import common

def main():
    df_train = common.load_train()
    X, y = df_train.loc[:, common.X_cols].values, df_train.target.values
    est = RandomForestClassifier(n_estimators=125)
    common.predict_and_report(est, X, y)

if __name__ == '__main__':
    main()


