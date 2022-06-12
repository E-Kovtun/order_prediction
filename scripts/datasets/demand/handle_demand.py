import sys
import pandas as pd
from utils.split_data import split_dataset
import os


def prepare_demand_dataset(init_folder, prepared_folder):
    data = pd.read_csv(os.path.join(init_folder, 'Historical Product Demand.csv'))
    data = data[['Warehouse', 'Product_Category', 'Date', 'Order_Demand']]
    data['Date'] = pd.to_datetime(data['Date'], format="%Y/%m/%d")
    nat_indices = []
    for ind in range(data.shape[0]):
        if pd.isnull(data.loc[ind, 'Date']):
            nat_indices.append(ind)
    data.drop(nat_indices, axis=0, inplace=True)
    data.reset_index(inplace=True)
    data['Product_Category'] = data['Product_Category'].apply(lambda x: int(x.split('_')[1].lstrip('0')))
    id_dict = {'Whse_J': 0, 'Whse_S': 1, 'Whse_C': 2, 'Whse_A': 3}
    data['Warehouse'] = data['Warehouse'].apply(lambda x: id_dict[x])
    data['Order_Demand'] = data['Order_Demand'].apply(lambda x: int(x.replace("(", "").replace(")", "")))
    data.rename(columns={'Warehouse': 'id', 'Date': 'date', 'Product_Category': 'label', 'Order_Demand': 'amount'}, inplace=True)
    split_dataset(data, prepared_folder)


if __name__ == "__main__":
    init_folder = sys.argv[1]
    prepared_folder = sys.argv[2]
    prepare_demand_dataset(init_folder, prepared_folder)
