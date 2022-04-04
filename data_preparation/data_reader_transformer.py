from data_preparation.dataset_preparation_transformer import OrderDataset
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import torch
from torch.nn.functional import pad

class OrderReader(Dataset):
    def __init__(self, data_folder, train_file, test_file, valid_file, look_back, phase):
        super(OrderReader, self).__init__()
        self.order_dataset = OrderDataset(data_folder, train_file, test_file, valid_file, look_back)
        self.phase = phase

        [train_final, test_final, valid_final], cat_vocab_size, id_vocab_size, amount_vocab_size, dt_vocab_size, \
            max_cat_len, dt_encoder = self.order_dataset.preprocess_dataframe()

        if self.phase == 'train':
            self.df_final, self.ind_combinations = train_final, self.order_dataset.window_combinations(train_final)
        if self.phase == 'test':
            self.df_final, self.ind_combinations = test_final, self.order_dataset.window_combinations(test_final)
        if self.phase == 'valid':
            self.df_final, self.ind_combinations = valid_final, self.order_dataset.window_combinations(valid_final)

        self.cat_vocab_size = cat_vocab_size
        self.id_vocab_size = id_vocab_size
        self.amount_vocab_size = amount_vocab_size
        self.dt_vocab_size = dt_vocab_size
        self.dt_encoder = dt_encoder

        self.max_cat_len = max_cat_len

    def __len__(self):
        return len(self.ind_combinations)

    def __getitem__(self, index):
        # look_back x max_cat_len
        cat_arr = torch.stack([pad(input=torch.tensor(self.df_final.loc[ref_i, self.order_dataset.categorical], dtype=torch.int64),
                                   pad=(0, self.max_cat_len - len(self.df_final.loc[ref_i, self.order_dataset.categorical])),
                                   mode='constant',
                                   value=self.cat_vocab_size)
                               for ref_i in self.ind_combinations[index][0]], dim=0)

        # max_cat_len
        current_cat = pad(input=torch.tensor(self.df_final.loc[self.ind_combinations[index][1], self.order_dataset.categorical], dtype=torch.int64),
                          pad=(0, self.max_cat_len - len(self.df_final.loc[self.ind_combinations[index][1], self.order_dataset.categorical])),
                          mode='constant',
                          value=self.cat_vocab_size)

        # look_back
        # dt_arr = torch.tensor([self.dt_encoder.fit_transform(np.array(self.df_final.loc[self.ind_combinations[index][1], self.order_dataset.dt] -
        #                        self.df_final.loc[ref_i, self.order_dataset.dt]).reshape(-1, 1))[0][0]
        #                        for ref_i in self.ind_combinations[index][0]], dtype=torch.int64)

        # look_back
        dt_arr = torch.tensor([self.dt_encoder.fit_transform(np.array(self.df_final.loc[ref_i, self.order_dataset.dt]).reshape(-1, 1))[0][0]
                               for ref_i in self.ind_combinations[index][0]], dtype=torch.int64)

        # look_back x max_cat_len
        amount_arr = torch.stack([pad(input=torch.tensor(self.df_final.loc[ref_i, self.order_dataset.amount], dtype=torch.int64),
                                      pad=(0, self.max_cat_len - len(self.df_final.loc[ref_i, self.order_dataset.amount])),
                                      mode='constant',
                                      value=self.amount_vocab_size)
                                  for ref_i in self.ind_combinations[index][0]], dim=0)

        id_arr = torch.tensor(self.df_final.loc[self.ind_combinations[index][0][0], self.order_dataset.id],
                              dtype=torch.int64)

        return cat_arr, current_cat, dt_arr, amount_arr, id_arr

        # number_arr = (torch.tensor(len(self.df_final.loc[self.ind_combinations[index][1], self.order_dataset.categorical]), dtype=torch.int64) - 1)
        #
        # return cat_arr, current_cat, dt_arr, amount_arr, id_arr, number_arr
