import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from models.regression.lstm_simultaneous import DoubleVariableNet
from data_preparation.data_reader_transactions import OrderReader
from data_preparation.dataset_preparation_upd import OrderDataset
from sklearn.metrics import r2_score, mean_absolute_percentage_error, accuracy_score
import numpy as np
from sacred import Experiment
import os
from utils.earlystopping import EarlyStopping
import json
from catalyst import metrics
from tqdm import tqdm
from utils.utils import own_r2_metric
torch.manual_seed(2)

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


def multilabel_crossentropy_loss(output, multilabel_onehot_target):
    # output (logits) - batch_size x cat_vocab_size
    # multilabel_onehot_target - batch_size x cat_vocab_size + 1
    multi_loss = torch.sum(torch.stack([torch.sum(torch.stack([cross_entropy(output[b, :].reshape(1, -1), label.reshape(-1))
                                         for label in torch.where(multilabel_onehot_target[b, :] == 1)[0]], dim=0))
                                         for b in range(output.shape[0])]), dim=0)
    return multi_loss


def smooth_pairwise_loss(output, multilabel_onehot_target):
    multilabel_onehot_target = multilabel_onehot_target[:, :-1]
    smooth_pairwise_loss = torch.sum(torch.stack([torch.log(torch.tensor([1.], requires_grad=True) + torch.sum(torch.stack([torch.exp(output[b, i0] - output[b, i1])
                            for i0 in torch.where(multilabel_onehot_target[b, :] == 0)[0][torch.randperm(len(torch.where(multilabel_onehot_target[b, :] == 0)[0]))[:10]].tolist()
                            for i1 in torch.where(multilabel_onehot_target[b, :] == 1)[0][torch.randperm(len(torch.where(multilabel_onehot_target[b, :] == 1)[0]))[:10]].tolist()])))
                            for b in range(output.shape[0])]))
    return smooth_pairwise_loss


def pairwise_loss(output, multilabel_onehot_target):
    multilabel_onehot_target = multilabel_onehot_target[:, :-1]
    pairwise_loss = torch.sum(torch.stack([torch.sum(torch.stack([torch.maximum(torch.tensor(0.), torch.tensor(1.) + output[b, i0] - output[b, i1])
                  for i0 in torch.where(multilabel_onehot_target[b, :] == 0)[0][torch.randperm(len(torch.where(multilabel_onehot_target[b, :] == 0)[0]))[:10]].tolist()
                  for i1 in torch.where(multilabel_onehot_target[b, :] == 1)[0][torch.randperm(len(torch.where(multilabel_onehot_target[b, :] == 1)[0]))[:10]].tolist()]))
                  for b in range(output.shape[0])]))
    return pairwise_loss


def mean_patk(output, multilabel_onehot_target, k):
    mean_patk_metric = np.mean([len(np.intersect1d(torch.topk(output[b, :], k=k, dim=0).indices.numpy(),
                                                   torch.where(multilabel_onehot_target[b, :] == 1)[0].numpy())) / k
                                for b in range(output.shape[0])])
    return mean_patk_metric


def mean_ratk(output, multilabel_onehot_target, k):
    mean_ratk_metric = np.mean([len(np.intersect1d(torch.topk(output[b, :], k=k, dim=0).indices.numpy(),
                                                   torch.where(multilabel_onehot_target[b, :] == 1)[0].numpy())) /
                                len(torch.where(multilabel_onehot_target[b, :] == 1)[0].numpy())
                                for b in range(output.shape[0])])
    return mean_ratk_metric


def mapk(output, multilabel_onehot_target, k):
    mapk_metric = np.mean([np.mean([len(np.intersect1d(torch.topk(output[b, :], k=i, dim=0).indices.numpy(),
                                                      torch.where(multilabel_onehot_target[b, :] == 1)[0].numpy())) / i
                                    for i in range(1, k+1)]) for b in range(output.shape[0])])
    return mapk_metric


def train():

    data_folder = "../initial_data/"
    # train_file = "df_beer_train_nn.csv"
    # test_file = "df_beer_test.csv"
    # valid_file = "df_beer_valid_nn.csv"
    train_file = "tr_train.csv"
    test_file = "tr_test.csv"
    valid_file = "tr_valid.csv"
    look_back = 3

    num_epochs = 500
    batch_size = 16
    dataloader_num_workers = 2

    optimizer_lr = 1e-3

    scheduler_factor = 0.3
    scheduler_patience = 5

    early_stopping_patience = 15

    alpha = 1e-4
    model_name = 'LSTM_simultaneous'
    results_folder = f'../results_transactions/{model_name}/'
    checkpoint = results_folder + f'checkpoints/look_back_{look_back}.pt'

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    train_dataset = OrderReader(data_folder, train_file, test_file, valid_file, look_back, 'train')
    test_dataset = OrderReader(data_folder, train_file, test_file, valid_file, look_back, 'test')
    valid_dataset = OrderReader(data_folder, train_file, test_file, valid_file, look_back, 'valid')

    linear_num_feat_dim = 32
    cat_embedding_dim = 256
    lstm_hidden_dim = 128
    cat_vocab_size = train_dataset.cat_vocab_size
    id_vocab_size = train_dataset.id_vocab_size
    id_embedding_dim = 128
    linear_concat1_dim = 128
    linear_concat2_dim = 64

    net = DoubleVariableNet(linear_num_feat_dim, cat_embedding_dim, lstm_hidden_dim,
                            cat_vocab_size, id_vocab_size,
                            id_embedding_dim, linear_concat1_dim, linear_concat2_dim).to(device)

    regr_loss = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(net.parameters(), lr=optimizer_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=dataloader_num_workers)

    os.makedirs(results_folder+'checkpoints/', exist_ok=True)
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True, path=checkpoint)

    for epoch in range(1, num_epochs+1):
        net.train(True)
        epoch_train_loss = 0
        print('Training...')
        for batch_ind, batch_arrays in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            batch_arrays = [arr.to(device) for arr in batch_arrays]
            [batch_cat_arr, batch_mask_cat,
             batch_current_cat, batch_mask_current_cat, batch_onehot_current_cat,
             batch_num_arr, batch_id_arr, batch_target_amount, batch_target_cat] = batch_arrays
            # print(batch_cat_arr.shape)
            # print(batch_mask_cat.shape)
            # print(batch_current_cat.shape)
            # print(batch_mask_current_cat.shape)
            # print(batch_onehot_current_cat.shape)
            # print(batch_num_arr.shape)
            # print(batch_id_arr.shape)
            # print(batch_target_amount.shape)
            optimizer.zero_grad()
            output_amount, output_material = net(batch_cat_arr, batch_mask_cat, batch_num_arr, batch_id_arr)

            nonzero_indices = batch_onehot_current_cat.reshape(-1).nonzero()
            batch_predicted_amounts = (output_amount * batch_onehot_current_cat).reshape(-1)[nonzero_indices]
            batch_gt_amounts = batch_target_amount.reshape(-1)[(batch_target_amount.reshape(-1) - train_dataset.amount_padding_value).nonzero()]

            loss = regr_loss(batch_predicted_amounts, batch_gt_amounts) + \
                   alpha * multilabel_crossentropy_loss(output_material, batch_onehot_current_cat)
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}/{num_epochs} || Train loss {epoch_train_loss}')

        print('Validation...')
        net.train(False)
        epoch_valid_loss = 0
        for batch_ind, batch_arrays in enumerate(valid_dataloader):
            batch_arrays = [arr.to(device) for arr in batch_arrays]
            [batch_cat_arr, batch_mask_cat,
             batch_current_cat, batch_mask_current_cat, batch_onehot_current_cat,
             batch_num_arr, batch_id_arr, batch_target_amount, batch_target_cat] = batch_arrays
            output_amount, output_material = net(batch_cat_arr, batch_mask_cat, batch_num_arr, batch_id_arr)

            nonzero_indices = batch_onehot_current_cat.reshape(-1).nonzero()
            batch_predicted_amounts = (output_amount * batch_onehot_current_cat).reshape(-1)[nonzero_indices]
            batch_gt_amounts = batch_target_amount.reshape(-1)[(batch_target_amount.reshape(-1) - train_dataset.amount_padding_value).nonzero()]

            # loss = regr_loss(batch_predicted_amounts, batch_gt_amounts) + \
            #        alpha * multilabel_crossentropy_loss(output_material, batch_onehot_current_cat)
            loss = 10**6 * regr_loss(batch_predicted_amounts, batch_gt_amounts) + \
                   multilabel_crossentropy_loss(output_material, batch_onehot_current_cat)
            epoch_valid_loss += loss.item()

        print(f'Epoch {epoch}/{num_epochs} || Valid loss {epoch_valid_loss}')

        scheduler.step(epoch_valid_loss)

        early_stopping(epoch_valid_loss, net)
        if early_stopping.early_stop:
            print('Early stopping')
            break

# #----------------------------------------------
    net = DoubleVariableNet(linear_num_feat_dim, cat_embedding_dim, lstm_hidden_dim,
                            cat_vocab_size, id_vocab_size,
                            id_embedding_dim, linear_concat1_dim, linear_concat2_dim).to(device)
    net.load_state_dict(torch.load(checkpoint, map_location=device))
    net.train(False)
    print('Testing...')
    output_list = []
    gt_list = []
    for batch_ind, batch_arrays in enumerate(test_dataloader):
        batch_arrays = [arr.to(device) for arr in batch_arrays]
        [batch_cat_arr, batch_mask_cat,
         batch_current_cat, batch_mask_current_cat, batch_onehot_current_cat,
         batch_num_arr, batch_id_arr, batch_target_amount, batch_target_cat] = batch_arrays
        output_amount, output_material = net(batch_cat_arr, batch_mask_cat, batch_num_arr, batch_id_arr)

        output_list.append(output_material.detach().cpu())
        gt_list.append(batch_onehot_current_cat.detach().cpu())

    all_output = torch.cat(output_list, dim=0)
    all_gt = torch.cat(gt_list, dim=0)
    test_mean_patk = {i: mean_patk(all_output, all_gt, k=i) for i in range(1, 5)}
    test_mean_ratk = {i: mean_ratk(all_output, all_gt, k=i) for i in range(1, 5)}
    test_mapk = {i: mapk(all_output, all_gt, k=i) for i in range(1, 5)}

    print(f'Test Precision@k {test_mean_patk} || Test Recall@k {test_mean_ratk}|| MAP@k {test_mapk})')

    with open(results_folder + f'{os.path.splitext(os.path.basename(checkpoint))[0]}.json', 'w', encoding='utf-8') as f:
        json.dump({'test_patk': test_mean_patk, 'test_ratk': test_mean_ratk, 'test_mapk': test_mapk}, f)


if __name__ == '__main__':
    train()
