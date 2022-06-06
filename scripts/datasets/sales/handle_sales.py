import sys
import pandas as pd
from utils.split_data import split_dataset
import os


def prepare_sales_dataset(init_folder, prepared_folder):
    data0 = pd.read_csv(os.path.join(init_folder, 'sales_train.csv'))
    items = pd.read_csv(os.path.join(init_folder, 'items.csv'))
    data0['item'] = data0['item_id'].apply(lambda x: items.loc[x, 'item_category_id'])
    data = data0.loc[data0['item_cnt_day'] > 0, ['date', 'shop_id', 'item_cnt_day', 'item']].reset_index(drop=True)
    data['date'] = pd.to_datetime(data['date'], format="%d.%m.%Y")
    data.rename(columns={'shop_id': 'id', 'date': 'date', 'item': 'label', 'item_cnt_day': 'amount'}, inplace=True)
    split_dataset(data, prepared_folder)


if __name__ == "__main__":
    init_folder = sys.argv[1]
    prepared_folder = sys.argv[2]
    prepare_sales_dataset(init_folder, prepared_folder)
