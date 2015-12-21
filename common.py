import pandas as pd

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
