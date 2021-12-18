from data_preparation.dataset_preparation import OrderDataset
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm

class OrderReader(Dataset):
    def __init__(self, data_folder, train_file, test_file, look_back, fix_material, current_info, predicted_value, phase):
        super(OrderReader, self).__init__()
        self.order_dataset = OrderDataset(data_folder, train_file, test_file, look_back, fix_material, current_info, predicted_value)
        self.const_features = ['Ship.to', 'PLZ', 'Material']
        self.changing_features = ['dt', 'Amount_HL']
        self.phase = phase # for now only train and test

        train_grouped, test_grouped, mms_amount = self.prepare_input_data()
        self.mms_amount = mms_amount
        if self.phase == 'train':
            self.df_grouped, self.ind_combinations = train_grouped, self.order_dataset.window_combinations(train_grouped)
        if self.phase == 'test':
            self.df_grouped, self.ind_combinations = test_grouped, self.order_dataset.window_combinations(test_grouped)

        self.const_arr, self.changing_arr, self.target = self.arrange_matrix()

    def prepare_input_data(self):
        train_grouped, test_grouped = self.order_dataset.preprocess_dataframe()
        mms_dt = MinMaxScaler()
        train_grouped.loc[:, 'dt'] = mms_dt.fit_transform(train_grouped['dt'].values[:, None])
        test_grouped.loc[:, 'dt'] = mms_dt.transform(test_grouped['dt'].values[:, None])
        mms_amount = MinMaxScaler()
        train_grouped.loc[:, 'Amount_HL'] = mms_amount.fit_transform(train_grouped['Amount_HL'].values[:, None])
        test_grouped.loc[:, 'Amount_HL'] = mms_amount.transform(test_grouped['Amount_HL'].values[:, None])
        return train_grouped, test_grouped, mms_amount

    def arrange_matrix(self):
        const_arr = np.array([self.df_grouped.loc[ref_points[0], self.const_features].values.tolist()
                              for ref_points, pred_point in tqdm(self.ind_combinations)])
        changing_arr = np.array([self.df_grouped.loc[ref_points, self.changing_features].values.tolist()
                                 for ref_points, pred_point in tqdm(self.ind_combinations)])
        target = np.array([self.df_grouped.loc[pred_point, self.order_dataset.target_feature]
                          for ref_points, pred_point in tqdm(self.ind_combinations)])[:, None]
        return const_arr, changing_arr, target

    def __len__(self):
        return len(self.ind_combinations)

    def __getitem__(self, index):
        return self.const_arr[index, :], self.changing_arr[index, :, :], self.target[index]



# if __name__ == '__main__':
#         data_folder = "../initial_data/"
#         train_file = "df_beer_train.csv"
#         test_file = "df_beer_test.csv"
#         look_back = 2
#         fix_material = True
#         current_info = False
#         predicted_value = "amount"
#         phase = "train"
#
#         # order_reader = OrderReader(data_folder, train_file, test_file, look_back, fix_material, current_info, predicted_value, phase)
#         # const_arr, changing_arr, target = order_reader.arrange_matrix()
#         # print(const_arr.shape, changing_arr.shape, target.shape)
#
#         train_dataset = OrderReader(data_folder, train_file, test_file, look_back, fix_material, current_info, predicted_value, phase)
#         const_ex, changing_ex, target_ex = train_dataset[0]
#         print(const_ex.shape, changing_ex.shape, target_ex.shape)
#         print('----------------------')
#         print(const_ex, changing_ex, target_ex)
