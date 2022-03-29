import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
import torch


def get_vocab(df_train, df_test, df_valid, feature_name):
    vocab = np.sort(np.unique(list(np.unique(df_train[feature_name])) +
                              list(np.unique(df_test[feature_name])) +
                              list(np.unique(df_valid[feature_name])))).reshape(1, -1)
    return vocab


def get_fitted_scaler(df_train, df_test, df_valid, feature_name):
    mms = MinMaxScaler().fit(np.array((list(df_train[feature_name].values) +
                                       list(df_test[feature_name].values) +
                                       list(df_valid[feature_name].values))).reshape(-1, 1))
    return mms


def get_max_cat_len(df_train, df_test, df_valid, cat_feature):
    history_lengths = pd.concat([df_train[cat_feature], df_valid[cat_feature], df_test[cat_feature]], axis=0).apply(lambda x: len(x))
    return np.max(history_lengths)


def own_r2_metric(output, target):
    target_mean = torch.mean(target)
    ss_res = torch.sum((target - output) ** 2)
    ss_tot = torch.sum((target - target_mean) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def get_fitted_discretizer(df_train, df_test, df_valid, amount_feature):
    all_amounts = pd.concat([df_train[amount_feature], df_test[amount_feature], df_valid[amount_feature]]).values
    amount_discretizer = KBinsDiscretizer(n_bins=9, encode='ordinal', strategy='quantile')
    amount_discretizer.fit(all_amounts.reshape(-1, 1))
    return amount_discretizer


def get_max_dt(df_train, df_test, df_valid, dt_fetaure):
    all_dt = pd.concat([df_train[dt_fetaure], df_test[dt_fetaure], df_valid[dt_fetaure]]).values
    return np.max(all_dt).astype(np.int64)

