import numpy as np
import os
from tqdm import tqdm


def split_dataset(data, prepared_folder, r_train=0.70, r_valid=0.17):
    data_reord = data.sort_values(by='date', axis=0)
    min_history_len = 7
    date_dict = {}
    for id_num in tqdm(data_reord['id'].unique()):
        unique_dates = np.unique(data_reord.loc[data_reord['id'] == id_num, 'date'].values)
        if len(unique_dates) >= min_history_len:
            border_date0 = unique_dates[int(r_train * len(unique_dates))]
            border_date1 = unique_dates[int((r_train + r_valid) * len(unique_dates))]
            date_dict[id_num] = [border_date0, border_date1]

    train_indices = []
    valid_indices = []
    test_indices = []
    for id_num in tqdm(date_dict.keys()):
        id_indices = data.loc[data['id'] == id_num].index
        for ind in id_indices:
            if data.loc[ind, 'date'] <= date_dict[id_num][0]:
                train_indices.append(ind)
            if (data.loc[ind, 'date'] > date_dict[id_num][0]) and \
                    (data.loc[ind, 'date'] <= date_dict[id_num][1]):
                valid_indices.append(ind)
            if data.loc[ind, 'date'] > date_dict[id_num][1]:
                test_indices.append(ind)

    data.iloc[train_indices, :].to_csv(os.path.join(prepared_folder, 'train.csv'))
    data.iloc[valid_indices, :].to_csv(os.path.join(prepared_folder, 'valid.csv'))
    data.iloc[test_indices, :].to_csv(os.path.join(prepared_folder, 'test.csv'))

