from sklearn.ensemble import GradientBoostingClassifier

import common

def main():
    df_train = common.load_train()
    X, y = df_train.loc[:, common.X_cols].values, df_train.target.values
    est = GradientBoostingClassifier(n_estimators=35, subsample=0.7, max_features=0.7, max_depth=4)
    common.predict_and_report(est, X, y)

if __name__ == '__main__':
    main()



