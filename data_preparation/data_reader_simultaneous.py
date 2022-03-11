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
        self.num_arr, self.id_arr, self.target_amount, self.target_cat = self.arrange_matrix()

    def arrange_matrix(self):
        # full_data_len x look_back x max_day_len0
        cat_arr = pad_sequence([pad_sequence([torch.tensor(self.df_final.loc[ref_i, self.order_dataset.categorical], dtype=torch.int64).unsqueeze(1)
                                for ref_i in ref_points], padding_value=self.cat_padding_value).squeeze(2)
                  for ref_points, pred_point in self.ind_combinations], padding_value=self.cat_padding_value).transpose(1, 0).transpose(2, 1)

        # full_data_len x look_back x max_day_len0
        mask_cat = torch.tensor(~(cat_arr == self.cat_padding_value), dtype=torch.int64)

        # full_data_len x max_day_len1
        current_cat = pad_sequence([torch.tensor(self.df_final.loc[pred_point, self.order_dataset.categorical], dtype=torch.int64).unsqueeze(1)
                      for ref_points, pred_point in self.ind_combinations], padding_value=self.cat_padding_value).squeeze(2).transpose(1, 0)

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
        target_amount = pad_sequence([torch.tensor(self.df_final.loc[pred_point, self.order_dataset.amount], dtype=torch.float32).unsqueeze(1)
                 for ref_points, pred_point in self.ind_combinations], padding_value=self.amount_padding_value).squeeze(2).transpose(1, 0)

        # Target for classification task
        # full_data_len x 1
        target_cat = (torch.tensor([len(self.df_final.loc[pred_point, self.order_dataset.categorical])
                               for ref_points, pred_point in self.ind_combinations], dtype=torch.int64) - 1).unsqueeze(1)

        return cat_arr, mask_cat, current_cat, mask_current_cat, onehot_current_cat, num_arr, id_arr, target_amount, target_cat

    def __len__(self):
        return len(self.ind_combinations)

    def __getitem__(self, index):
        return [self.cat_arr[index, :, :], self.mask_cat[index, :, :],
                self.current_cat[index, :], self.mask_current_cat[index, :, :], self.onehot_current_cat[index, :],
                self.num_arr[index, :, :], self.id_arr[index, :], self.target_amount[index, :], self.target_cat[index, :]]

