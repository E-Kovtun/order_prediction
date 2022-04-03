from data_preparation.dataset_preparation_upd import OrderDataset
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import one_hot

class OrderReader(Dataset):
    def __init__(self, data_folder, train_file, test_file, valid_file, look_back, phase):
        super(OrderReader, self).__init__()
        self.order_dataset = OrderDataset(data_folder, train_file, test_file, valid_file, look_back)
        self.phase = phase

        [train_final, test_final, valid_final], mms, cat_vocab_size, id_vocab_size = \
            self.order_dataset.preprocess_dataframe()
        self.mms = mms
        if self.phase == 'train':
            self.df_final, self.ind_combinations = train_final, self.order_dataset.window_combinations(train_final)
        if self.phase == 'test':
            self.df_final, self.ind_combinations = test_final, self.order_dataset.window_combinations(test_final)
        if self.phase == 'valid':
            self.df_final, self.ind_combinations = valid_final, self.order_dataset.window_combinations(valid_final)

        self.cat_vocab_size = cat_vocab_size
        self.id_vocab_size = id_vocab_size
        self.cat_padding_value = cat_vocab_size
        self.amount_padding_value = 100

        # for now we sum embeddings and amounts relate to one day
        # num_arr consists of summed amount and dt
        self.cat_arr, self.mask_cat, \
        self.current_cat, self.mask_current_cat, self.onehot_current_cat, \
        self.num_arr, self.id_arr, self.target = self.arrange_matrix() #, self.current_minus1_cat = self.arrange_matrix()

    def arrange_matrix(self):
        # full_data_len x look_back x max_day_len0
        cat_arr = pad_sequence([pad_sequence([torch.tensor(self.df_final.loc[ref_i, self.order_dataset.categorical], dtype=torch.int64).unsqueeze(1)
                                for ref_i in ref_points], padding_value=self.cat_padding_value).squeeze()
                  for ref_points, pred_point in self.ind_combinations], padding_value=self.cat_padding_value).transpose(1, 0).transpose(2, 1)

        # full_data_len x look_back x max_day_len0
        mask_cat = torch.tensor(~(cat_arr == self.cat_padding_value), dtype=torch.int64)

        # TODO current t - 1 experiment
        # full_data_len x max_day_len1
        #current_minus1_cat = pad_sequence([torch.tensor(self.df_final.loc[pred_point-1, self.order_dataset.categorical], dtype=torch.int64).unsqueeze(1)
        #              for ref_points, pred_point in self.ind_combinations], padding_value=self.cat_padding_value).squeeze().transpose(1, 0)
        #mask_current_cat = torch.tensor(~(current_minus1_cat == self.cat_padding_value), dtype=torch.int64).unsqueeze(2)
        #current_minus1_cat = torch.sum(one_hot(current_minus1_cat, num_classes=self.cat_vocab_size+1) * mask_current_cat, dim=1)

        current_cat = pad_sequence([torch.tensor(self.df_final.loc[pred_point, self.order_dataset.categorical], dtype=torch.int64).unsqueeze(1)
                      for ref_points, pred_point in self.ind_combinations], padding_value=self.cat_padding_value).squeeze().transpose(1, 0)

        # full_data_len x max_day_len1 x 1
        mask_current_cat = torch.tensor(~(current_cat == self.cat_padding_value), dtype=torch.int64).unsqueeze(2)

        # full_data_len x cat_vocab_size+1
        onehot_current_cat = torch.sum(one_hot(current_cat, num_classes=self.cat_vocab_size+1) * mask_current_cat, dim=1)

        # full_data_len x look_back x 2
        num_arr = torch.stack((torch.tensor([[np.sum(self.df_final.loc[ref_i, self.order_dataset.amount]) for ref_i in ref_points]
                                             for ref_points, pred_point in self.ind_combinations], dtype=torch.float32),
                               torch.tensor([self.df_final.loc[ref_points, self.order_dataset.dt].values.tolist()
                                            for ref_points, pred_point in self.ind_combinations], dtype=torch.float32)), dim=2)

        # full_data_len x 1
        id_arr = torch.tensor([self.df_final.loc[ref_points[0], self.order_dataset.id]
                              for ref_points, pred_point in self.ind_combinations], dtype=torch.int64).reshape(-1, 1)

        # full_data_len x max_day_len1
        # amount_arr = pad_sequence([torch.tensor(self.df_final.loc[pred_point, self.order_dataset.amount], dtype=torch.float32).unsqueeze(1)
        #              for ref_points, pred_point in self.ind_combinations], padding_value=self.amount_padding_value).squeeze().transpose(1, 0)

        # full_data_len x cat_vocab_size+1
        # target = torch.sum(one_hot(current_cat, num_classes=self.cat_vocab_size+1) * mask_current_cat * amount_arr.unsqueeze(2), dim=1)

        # # num of all amounts in full_data_len
        # target = torch.tensor([[a for ref_points, pred_point in self.ind_combinations
        #                           for a in self.df_final.loc[pred_point, self.order_dataset.amount]]], dtype=torch.float32)


        # full_data_len x max_day_len1
        target = pad_sequence([torch.tensor(self.df_final.loc[pred_point, self.order_dataset.amount], dtype=torch.float32).unsqueeze(1)
                 for ref_points, pred_point in self.ind_combinations], padding_value=self.amount_padding_value).squeeze().transpose(1, 0)

        return cat_arr, mask_cat, current_cat, mask_current_cat, onehot_current_cat, num_arr, id_arr, target #, current_minus1_cat

    def __len__(self):
        return len(self.ind_combinations)

    def __getitem__(self, index):
        return [self.cat_arr[index, :, :], self.mask_cat[index, :, :],
                self.current_cat[index, :], self.mask_current_cat[index, :, :], self.onehot_current_cat[index, :],
                self.num_arr[index, :, :], self.id_arr[index, :], self.target[index, :]] #, self.current_minus1_cat[index, :]]



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
