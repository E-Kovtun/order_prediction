import sys
import pandas as pd
import numpy as np
import os


def prepare_order_dataset(init_folder, prepared_folder):
    train = pd.read_csv(init_folder[:-1]+'df_beer_train.csv', usecols=['Ship.to', 'Material', 'Delivery_Date_week', 'Amount_HL'])
    test = pd.read_csv(init_folder[:-1]+'df_beer_test.csv', usecols=['Ship.to', 'Material', 'Delivery_Date_week', 'Amount_HL'])

    train['Delivery_Date_week'] = pd.to_datetime(train['Delivery_Date_week'])
    train.rename(columns={'Ship.to': 'id', 'Delivery_Date_week': 'date', 'Material': 'label', 'Amount_HL': 'amount'}, inplace=True)
    test['Delivery_Date_week'] = pd.to_datetime(test['Delivery_Date_week'])
    test.rename(columns={'Ship.to': 'id', 'Delivery_Date_week': 'date', 'Material': 'label', 'Amount_HL': 'amount'}, inplace=True)

    data_reord = train.sort_values(by='date', axis=0)
    r = 0.8
    date_dict = {}
    for id_num in data_reord['id'].unique():
        unique_dates = np.unique(data_reord.loc[data_reord['id'] == id_num, 'date'].values)
        border_date = unique_dates[int(r * len(unique_dates))]
        date_dict[id_num] = border_date

    train_indices = []
    valid_indices = []
    for ind in train.index:
        id_num = train.loc[ind, 'id']
        if train.loc[ind, 'date'] <= date_dict[id_num]:
            train_indices.append(ind)
        else:
            valid_indices.append(ind)

    train.iloc[train_indices, :].to_csv(os.path.join(prepared_folder, 'train.csv'))
    train.iloc[valid_indices, :].to_csv(os.path.join(prepared_folder, 'valid.csv'))
    test.to_csv(os.path.join(prepared_folder, 'test.csv'))


if __name__ == "__main__":
    init_folder = sys.argv[1]
    prepared_folder = sys.argv[2]
    prepare_order_dataset(init_folder, prepared_folder)
