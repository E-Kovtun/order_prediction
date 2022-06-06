import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from models.regression.lstm_material import ClassificationNet
from data_preparation.data_reader_upd import OrderReader
from data_preparation.dataset_preparation_upd import OrderDataset
#from sklearn.metrics import r2_score, mean_absolute_percentage_error, accuracy_score
import numpy as np
from sacred import Experiment
import os
#from utils.earlystopping import EarlyStopping
import json
from catalyst import metrics
from tqdm import tqdm
#from utils.utils import own_r2_metric
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


# def smooth_pairwise_loss(output, multilabel_onehot_target):
#     # output (logits) - batch_size x cat_vocab_size
#     # multilabel_onehot_target - batch_size x cat_vocab_size + 1
#     multilabel_onehot_target = multilabel_onehot_target[:, :-1]
#     pairwise_loss = 0
#     for b in range(output.shape[0]):
#         zero_pos = torch.where(multilabel_onehot_target[b, :] == 0)[0].tolist()
#         one_pos = torch.where(multilabel_onehot_target[b, :] == 1)[0].tolist()
#         inner_sum = 0
#         for i0 in zero_pos:
#             for i1 in one_pos:
#                 inner_sum += torch.exp(output[b, i0] - output[b, i1])
#         pairwise_loss += torch.log(torch.tensor([1]) + inner_sum)
#     return pairwise_loss


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
    train_file = "df_beer_train_nn.csv"
    test_file = "df_beer_test.csv"
    valid_file = "df_beer_valid_nn.csv"
    look_back = 3

    num_epochs = 500
    batch_size = 128
    dataloader_num_workers = 2

    optimizer_lr = 1e-4

    scheduler_factor = 0.3
    scheduler_patience = 5

    early_stopping_patience = 15
    model_name = 'LSTM_material'
    results_folder = f'../results/{model_name}/'
    checkpoint = results_folder + f'checkpoints/look_back_{look_back}_pal_check.pt'

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

    net = ClassificationNet(linear_num_feat_dim, cat_embedding_dim, lstm_hidden_dim,
                            cat_vocab_size, id_vocab_size,
                            id_embedding_dim, linear_concat1_dim, linear_concat2_dim).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=optimizer_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience)
    # metric_mapk = metrics.MAPMetric(topk=[3])

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
             batch_num_arr, batch_id_arr, batch_target] = batch_arrays
            optimizer.zero_grad()
            output_material = net(batch_cat_arr, batch_mask_cat, batch_num_arr, batch_id_arr)

            # metric_mapk.reset()
            # metric_mapk.update(output_material, batch_onehot_current_cat[:, :-1])
            # train_mapk.append(metric_mapk.compute_key_value()['map03'])

            loss = multilabel_crossentropy_loss(output_material, batch_onehot_current_cat)
            # loss = smooth_pairwise_loss(output_material, batch_onehot_current_cat)
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
             batch_num_arr, batch_id_arr, batch_target] = batch_arrays
            output_material = net(batch_cat_arr, batch_mask_cat, batch_num_arr, batch_id_arr)

            # metric_mapk.reset()
            # metric_mapk.update(output_material, batch_onehot_current_cat[:, :-1])
            # valid_mapk.append(metric_mapk.compute_key_value()['map03'])

            loss = multilabel_crossentropy_loss(output_material, batch_onehot_current_cat)
            # loss = smooth_pairwise_loss(output_material, batch_onehot_current_cat)
            epoch_valid_loss += loss.item()

        print(f'Epoch {epoch}/{num_epochs} || Valid loss {epoch_valid_loss}')

        scheduler.step(epoch_valid_loss)

        early_stopping(epoch_valid_loss, net)
        if early_stopping.early_stop:
            print('Early stopping')
            break

# #----------------------------------------------
    net = ClassificationNet(linear_num_feat_dim, cat_embedding_dim, lstm_hidden_dim,
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
         batch_num_arr, batch_id_arr, batch_target] = batch_arrays
        output_material = net(batch_cat_arr, batch_mask_cat, batch_num_arr, batch_id_arr)
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

# @ex.automain
# def my_main():
#     train()

if __name__ == '__main__':
    train()
