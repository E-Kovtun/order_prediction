import sys
import pandas as pd
import numpy as np
from utils.split_data import split_dataset
import os


def prepare_liquor_dataset(init_folder, prepared_folder):
    data = pd.read_csv(os.path.join(init_folder, 'Iowa_Liquor_Sales.csv'),
                       usecols=['Date', 'Store Number', 'Category', 'Volume Sold (Liters)'])
    data = data.dropna()
    data.reset_index(inplace=True)
    data['Category'] = data['Category'].astype(np.int64)
    data['Date'] = pd.to_datetime(data['Date'], format="%m/%d/%Y")
    data.rename(columns={'Store Number': 'id', 'Date': 'date', 'Category': 'label', 'Volume Sold (Liters)': 'amount'}, inplace=True)
    split_dataset(data, prepared_folder)


if __name__ == "__main__":
    init_folder = sys.argv[1]
    prepared_folder = sys.argv[2]
    prepare_liquor_dataset(init_folder, prepared_folder)
