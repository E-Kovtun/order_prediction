import torch
from torch import nn
from torch.utils.data import DataLoader

import sys
sys.path.append('C:\\Users\\boeva\\Beer\\Repository', )

from models.regression.lstm_multidim_amount_upd import RegressionNet
from data_preparation.data_reader_upd import OrderReader
from data_preparation.dataset_preparation_upd import OrderDataset
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import numpy as np
from sacred import Experiment
import os
#from utils.earlystopping import EarlyStopping
import json
from tqdm import tqdm
from utils.utils import own_r2_metric

# ex = Experiment('LSTM_multidim_amount')

# @ex.config
# def config_train():
#     data_folder = "initial_data/"
#     train_file = "df_beer_train_nn.csv"
#     test_file = "df_beer_test.csv"
#     valid_file = "df_beer_valid_nn.csv"
#     look_back = 3
#     fix_material = False
#     current_info = False
#     predicted_value = "amount"

# @ex.automain
def train():

    data_folder = "C:\\Users\\boeva\\Beer\\"
    train_file = "df_beer_train_nn.csv"
    test_file = "df_beer_valid_nn.csv"
    valid_file = "df_beer_test_nn.csv"
    look_back = 3

    num_epochs = 100
    batch_size = 128
    dataloader_num_workers = 2

    optimizer_lr = 1e-4

    scheduler_factor = 0.3
    scheduler_patience = 5

    early_stopping_patience = 15
    model_name = 'LSTM_multidim_amount'
    results_folder = f'../results/{model_name}/'
    checkpoint = results_folder + 'checkpoints/look_back_{look_back}.pt'

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    train_dataset = OrderReader(data_folder, train_file, test_file, valid_file, look_back, 'train')
    test_dataset = OrderReader(data_folder, train_file, test_file, valid_file, look_back, 'test')
    valid_dataset = OrderReader(data_folder, train_file, test_file, valid_file, look_back, 'valid')

    linear_num_feat_dim = 16
    cat_embedding_dim = 256
    lstm_hidden_dim = 128
    cat_vocab_size = train_dataset.cat_vocab_size
    id_vocab_size = train_dataset.id_vocab_size
    id_embedding_dim = 128
    linear_concat1_dim = 128
    linear_concat2_dim = 64

    net = RegressionNet(linear_num_feat_dim, cat_embedding_dim, lstm_hidden_dim,
                        cat_vocab_size, id_vocab_size,
                        id_embedding_dim, linear_concat1_dim, linear_concat2_dim).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=optimizer_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience)
    regr_loss = nn.MSELoss(reduction='sum')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=dataloader_num_workers)

    os.makedirs(results_folder+'checkpoints/', exist_ok=True)
    #early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True, path=checkpoint)

    for epoch in range(1, num_epochs+1):
        net.train(True)
        epoch_train_loss = 0
        train_predicted_amounts = []
        train_gt_amounts = []
        print('Training...')
        for batch_ind, batch_arrays in enumerate(train_dataloader):
            batch_arrays = [arr.to(device) for arr in batch_arrays]
            [batch_cat_arr, batch_mask_cat,
             batch_current_cat, batch_mask_current_cat, batch_onehot_current_cat,
             batch_num_arr, batch_id_arr, batch_target] = batch_arrays
            optimizer.zero_grad()
            output, batch_onehot_current_cat, mask_current_cat = net(batch_cat_arr, batch_mask_cat,
                         batch_current_cat, batch_mask_current_cat, batch_onehot_current_cat,
                         batch_num_arr, batch_id_arr)
            output = output.reshape(-1)
            nonzero_indices = batch_onehot_current_cat.reshape(-1).nonzero()
            batch_predicted_amounts = output[nonzero_indices]

            batch_target = batch_target * mask_current_cat.reshape(-1, mask_current_cat.shape[1]) - (mask_current_cat.reshape(-1, mask_current_cat.shape[1]) - 1) * train_dataset.amount_padding_value
            batch_gt_amounts = batch_target.reshape(-1)[(batch_target.reshape(-1) - train_dataset.amount_padding_value).nonzero()]

            train_predicted_amounts.extend(batch_predicted_amounts.detach().cpu().tolist())
            train_gt_amounts.extend(batch_gt_amounts.detach().cpu().tolist())
            if batch_predicted_amounts.shape[0] != batch_gt_amounts.shape[0]:
                continue
            loss = regr_loss(batch_predicted_amounts, batch_gt_amounts)
            epoch_train_loss += loss.item()
            # loss = -own_r2_metric(batch_predicted_amounts, batch_gt_amounts)
            loss.backward()
            optimizer.step()

        # MAYBE TO USE IN LOSS ANF TRAIN R2 INVERSE TRANSFORM??
        # epoch_train_r2 = r2_score(np.array(train_gt_amounts, dtype=np.float64).reshape(-1, 1),
        #                           np.array(train_predicted_amounts, dtype=np.float64).reshape(-1, 1))
        epoch_train_r2 = r2_score(train_dataset.mms.inverse_transform(np.array(train_gt_amounts, dtype=np.float64).reshape(-1, 1)),
                                  train_dataset.mms.inverse_transform(np.array(train_predicted_amounts[:len(train_gt_amounts)], dtype=np.float64).reshape(-1, 1)))
        print(f'Epoch {epoch}/{num_epochs} || Train loss {epoch_train_loss} || Train r2_score {epoch_train_r2}')

        print('Validation...')
        net.train(False)
        epoch_valid_loss = 0
        valid_predicted_amounts = []
        valid_gt_amounts = []
        with torch.no_grad():
            for batch_ind, batch_arrays in enumerate(valid_dataloader):
                batch_arrays = [arr.to(device) for arr in batch_arrays]
                [batch_cat_arr, batch_mask_cat,
                 batch_current_cat, batch_mask_current_cat, batch_onehot_current_cat,
                 batch_num_arr, batch_id_arr, batch_target] = batch_arrays
                output, batch_onehot_current_cat, mask_current_cat = net(batch_cat_arr, batch_mask_cat,
                                                                         batch_current_cat, batch_mask_current_cat,
                                                                         batch_onehot_current_cat,
                                                                         batch_num_arr, batch_id_arr)
                output = output.reshape(-1)
                nonzero_indices = batch_onehot_current_cat.reshape(-1).nonzero()
                batch_predicted_amounts = output[nonzero_indices]

                batch_target = batch_target * mask_current_cat.reshape(-1, mask_current_cat.shape[1]) - (mask_current_cat.reshape(-1, mask_current_cat.shape[1]) - 1) * train_dataset.amount_padding_value
                batch_gt_amounts = batch_target.reshape(-1)[(batch_target.reshape(-1) - train_dataset.amount_padding_value).nonzero()]

                valid_predicted_amounts.extend(batch_predicted_amounts.detach().cpu().tolist())
                valid_gt_amounts.extend(batch_gt_amounts.detach().cpu().tolist())

                if batch_predicted_amounts.shape[0] != batch_gt_amounts.shape[0]:
                    continue
                loss = regr_loss(batch_predicted_amounts, batch_gt_amounts)
                epoch_valid_loss += loss.item()

        # MAYBE NOT USE INVERSE TRANSFORM
        # r2_metric_valid = r2_score(np.array(valid_gt_amounts, dtype=np.float64),
        #                            np.array(valid_predicted_amounts, dtype=np.float64))
        epoch_valid_r2 = r2_score(valid_dataset.mms.inverse_transform(np.array(valid_gt_amounts, dtype=np.float64).reshape(-1, 1)),
                                  valid_dataset.mms.inverse_transform(np.array(valid_predicted_amounts[:len(valid_gt_amounts)], dtype=np.float64).reshape(-1, 1)))
        print(f'Epoch {epoch}/{num_epochs} || Valid loss {epoch_valid_loss} || Valid r2_score {epoch_valid_r2}')
        scheduler.step(-epoch_valid_r2)

        #early_stopping(-epoch_valid_r2, net)
        #if early_stopping.early_stop:
        #    print('Early stopping')
        #    break

#----------------------------------------------
    #net = RegressionNet(linear_num_feat_dim, cat_embedding_dim, lstm_hidden_dim,
    #                    cat_vocab_size, id_vocab_size,
    #                    id_embedding_dim, linear_concat1_dim, linear_concat2_dim).to(device)
    #net.load_state_dict(torch.load(checkpoint, map_location=device))
    net.train(False)
    print('Testing...')
    test_predicted_amounts = []
    test_gt_amounts = []
    with torch.no_grad():
        for batch_ind, batch_arrays in enumerate(test_dataloader):
            batch_arrays = [arr.to(device) for arr in batch_arrays]
            [batch_cat_arr, batch_mask_cat,
             batch_current_cat, batch_mask_current_cat, batch_onehot_current_cat,
             batch_num_arr, batch_id_arr, batch_target] = batch_arrays
            output, batch_onehot_current_cat, mask_current_cat = net(batch_cat_arr, batch_mask_cat,
                                                                     batch_current_cat, batch_mask_current_cat,
                                                                     batch_onehot_current_cat,
                                                                     batch_num_arr, batch_id_arr)
            output = output.reshape(-1)
            nonzero_indices = batch_onehot_current_cat.reshape(-1).nonzero()
            batch_predicted_amounts = output[nonzero_indices]

            batch_target = batch_target * mask_current_cat.reshape(-1, mask_current_cat.shape[1]) - (
                        mask_current_cat.reshape(-1,
                                                 mask_current_cat.shape[1]) - 1) * train_dataset.amount_padding_value
            batch_gt_amounts = batch_target.reshape(-1)[
                (batch_target.reshape(-1) - train_dataset.amount_padding_value).nonzero()]

            if batch_predicted_amounts.shape[0] != batch_gt_amounts.shape[0]:
                continue

            test_predicted_amounts.extend(batch_predicted_amounts.detach().cpu().tolist())
            test_gt_amounts.extend(batch_gt_amounts.detach().cpu().tolist())

    # r2_metric_test = r2_score(np.array(test_gt_amounts, dtype=np.float64),
    #                           np.array(test_predicted_amounts, dtype=np.float64))

    r2_metric_test = r2_score(test_dataset.mms.inverse_transform(np.array(test_gt_amounts, dtype=np.float64).reshape(-1, 1)),
                              test_dataset.mms.inverse_transform(np.array(test_predicted_amounts, dtype=np.float64).reshape(-1, 1)))
    mape_test = mean_absolute_percentage_error(test_dataset.mms.inverse_transform(np.array(test_gt_amounts, dtype=np.float64).reshape(-1, 1)),
                              test_dataset.mms.inverse_transform(np.array(test_predicted_amounts, dtype=np.float64).reshape(-1, 1)))
    print(f'Test r2_score {r2_metric_test}')
    print(f'Test MAPE {mape_test}')
    with open(results_folder + f'look_back_{look_back}.json', 'w', encoding='utf-8') as f:
        json.dump({'test_r2_score': r2_metric_test, 'test_mape': mape_test}, f)

# @ex.automain
# def my_main():
#     train()

if __name__ == '__main__':
    train()
