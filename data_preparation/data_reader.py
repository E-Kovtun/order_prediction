from data_preparation.dataset_preparation import OrderDataset
from torch.utils.data import Dataset
import torch
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import one_hot


class OrderReader(Dataset):
    def __init__(self, prepared_folder, look_back, phase):
        super(OrderReader, self).__init__()
        self.order_dataset = OrderDataset(prepared_folder, look_back)
        self.phase = phase

        [train_final, test_final, valid_final], cat_vocab_size, id_vocab_size, amount_vocab_size, dt_vocab_size, \
        max_cat_len = self.order_dataset.preprocess_dataframe()

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
        self.max_cat_len = max_cat_len
        # FOR CLASS2C2AE
        # current_minus1_cat = pad_sequence([torch.tensor(
        #     self.df_final.loc[pred_point - 1, self.order_dataset.categorical], dtype=torch.int64).unsqueeze(1)
        #                                    for ref_points, pred_point in self.ind_combinations],
        #                                   padding_value=self.cat_vocab_size).squeeze().transpose(1, 0)
        #
        # mask_current_cat = torch.tensor(~(current_minus1_cat == self.cat_vocab_size), dtype=torch.int64).unsqueeze(2)
        # self.current_minus1_cat = torch.sum(
        #     one_hot(current_minus1_cat, num_classes=self.cat_vocab_size + 1) * mask_current_cat, dim=1)

    def __len__(self):
        return len(self.ind_combinations)

    def __getitem__(self, index):
        # look_back x max_cat_len
        cat_arr = torch.stack(
            [pad(input=torch.tensor(self.df_final.loc[ref_i, self.order_dataset.categorical], dtype=torch.int64),
                 pad=(0, self.max_cat_len - len(self.df_final.loc[ref_i, self.order_dataset.categorical])),
                 mode='constant',
                 value=self.cat_vocab_size)
             for ref_i in self.ind_combinations[index][0]], dim=0)

        # max_cat_len
        current_cat = pad(
            input=torch.tensor(self.df_final.loc[self.ind_combinations[index][1], self.order_dataset.categorical],
                               dtype=torch.int64),
            pad=(0, self.max_cat_len - len(
                self.df_final.loc[self.ind_combinations[index][1], self.order_dataset.categorical])),
            mode='constant',
            value=self.cat_vocab_size)

        dt_arr = torch.tensor([self.df_final.loc[ref_i, self.order_dataset.dt]
                               for ref_i in self.ind_combinations[index][0]], dtype=torch.int64)

        # look_back x max_cat_len
        amount_arr = torch.stack(
            [pad(input=torch.tensor(self.df_final.loc[ref_i, self.order_dataset.amount], dtype=torch.int64),
                 pad=(0, self.max_cat_len - len(self.df_final.loc[ref_i, self.order_dataset.amount])),
                 mode='constant',
                 value=self.amount_vocab_size)
             for ref_i in self.ind_combinations[index][0]], dim=0)

        id_arr = torch.tensor(self.df_final.loc[self.ind_combinations[index][0][0], self.order_dataset.id],
                              dtype=torch.int64)

        return cat_arr, current_cat, dt_arr, amount_arr, id_arr #,  self.current_minus1_cat[index, :]
