from data_preparation.dataset_preparation_upd import OrderDataset
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import one_hot, pad
from utils.utils import get_max_cat_len

class OrderReader(Dataset):
    def __init__(self, data_folder, train_file, test_file, valid_file, look_back, phase):
        super(OrderReader, self).__init__()
        self.order_dataset = OrderDataset(data_folder, train_file, test_file, valid_file, look_back)
        self.phase = phase

        [train_final, test_final, valid_final], mms, cat_vocab_size, id_vocab_size, max_cat_len = \
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

        self.max_cat_len = max_cat_len

        # current_minus1_cat = pad_sequence([torch.tensor(
        #     self.df_final.loc[pred_point - 1, self.order_dataset.categorical], dtype=torch.int64).unsqueeze(1)
        #                                    for ref_points, pred_point in self.ind_combinations],
        #                                   padding_value=self.cat_padding_value).squeeze().transpose(1, 0)
        #
        # mask_current_cat = torch.tensor(~(current_minus1_cat == self.cat_padding_value), dtype=torch.int64).unsqueeze(2)
        # self.current_minus1_cat = torch.sum(
        #     one_hot(current_minus1_cat, num_classes=self.cat_vocab_size + 1) * mask_current_cat, dim=1)

    def __len__(self):
        return len(self.ind_combinations)

    def __getitem__(self, index):
        # look_back x max_cat_len
        cat_arr = torch.stack([pad(input=torch.tensor(self.df_final.loc[ref_i, self.order_dataset.categorical], dtype=torch.int64),
                                   pad=(0, self.max_cat_len - len(self.df_final.loc[ref_i, self.order_dataset.categorical])),
                                   mode='constant',
                                   value=self.cat_padding_value)
                               for ref_i in self.ind_combinations[index][0]], dim=0)

        # look_back x max_cat_len
        mask_cat = torch.tensor(~(cat_arr == self.cat_padding_value), dtype=torch.int64)

        # max_cat_len
        current_cat = pad(input=torch.tensor(self.df_final.loc[self.ind_combinations[index][1], self.order_dataset.categorical], dtype=torch.int64),
                          pad=(0, self.max_cat_len - len(self.df_final.loc[self.ind_combinations[index][1], self.order_dataset.categorical])),
                          mode='constant',
                          value=self.cat_padding_value)

        # max_cat_len x 1
        mask_current_cat = torch.tensor(~(current_cat == self.cat_padding_value), dtype=torch.int64).unsqueeze(1)

        # cat_vocab_size+1
        onehot_current_cat = torch.sum(one_hot(current_cat, num_classes=self.cat_vocab_size + 1) * mask_current_cat, dim=0)

        # look_back x 2
        num_arr = torch.stack((torch.stack([torch.sum(torch.tensor(self.df_final.loc[ref_i, self.order_dataset.amount], dtype=torch.float32))
                                            for ref_i in self.ind_combinations[index][0]]),
                               torch.tensor(self.df_final.loc[self.ind_combinations[index][0], self.order_dataset.dt].values.tolist(), dtype=torch.float32)),
                               dim=1)

        #
        id_arr = torch.tensor(self.df_final.loc[self.ind_combinations[index][0][0], self.order_dataset.id],
                              dtype=torch.int64)

        # max_cat_len
        # target_amount = pad(input=torch.tensor(self.df_final.loc[self.ind_combinations[index][1], self.order_dataset.amount], dtype=torch.float32),
        #                     pad=(0, self.max_cat_len - len(self.df_final.loc[self.ind_combinations[index][1], self.order_dataset.amount])),
        #                     mode='constant',
        #                     value=self.amount_padding_value)

        #
        target_cat = (torch.tensor(len(self.df_final.loc[self.ind_combinations[index][1], self.order_dataset.categorical]), dtype=torch.int64) - 1)

        return cat_arr, mask_cat, current_cat, mask_current_cat, onehot_current_cat, num_arr, id_arr, target_cat#, self.current_minus1_cat[index, :]


