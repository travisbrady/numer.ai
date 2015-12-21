from sklearn.ensemble import RandomForestClassifier

import common

def main():
    df_train = common.load_train()
    X, y = df_train.loc[:, common.X_cols].values, df_train.target.values
    common.predict_and_report(RandomForestClassifier(n_estimators=50), X, y)

if __name__ == '__main__':
    main()


