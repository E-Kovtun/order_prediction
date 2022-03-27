import numpy as np
from sklearn.preprocessing import MinMaxScaler
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


def get_max_cat_len(df_final, cat_feature):
    history_lengths = []
    for i in range(len(df_final)):
        history_lengths.append(len(df_final.loc[i, cat_feature]))
    return np.max(history_lengths)


def own_r2_metric(output, target):
    target_mean = torch.mean(target)
    ss_res = torch.sum((target - output) ** 2)
    ss_tot = torch.sum((target - target_mean) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2