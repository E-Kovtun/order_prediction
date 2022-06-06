import sys
import pandas as pd
from utils.split_data import split_dataset
import os


def prepare_gender_dataset(init_folder, prepared_folder):
    data = pd.read_csv(os.path.join(init_folder, 'transactions.csv'))
    data = data[['customer_id', 'tr_datetime', 'tr_type', 'amount']]
    data['tr_datetime'] = data['tr_datetime'].apply(lambda x: int(x.split(' ')[0]))
    data['tr_datetime'] = data['tr_datetime'].apply(lambda x: x // 7)
    data['amount'] = abs(data['amount'])
    data.rename(columns={'customer_id': 'id', 'tr_datetime': 'date', 'tr_type': 'label', 'amount': 'amount'}, inplace=True)
    split_dataset(data, prepared_folder)


if __name__ == "__main__":
    init_folder = sys.argv[1]
    prepared_folder = sys.argv[2]
    prepare_gender_dataset(init_folder, prepared_folder)
