import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy, one_hot
from models.regression.transformer1 import TransformerNet
from data_preparation.data_reader_transformer import OrderReader
import numpy as np
import os
from utils.earlystopping import EarlyStopping
import json
from tqdm import tqdm
torch.manual_seed(2)


def multilabel_crossentropy_loss(output, multilabel_onehot_target):
    # output (logits) - batch_size x cat_vocab_size
    # multilabel_onehot_target - batch_size x cat_vocab_size + 1
    multi_loss = torch.sum(torch.stack([torch.sum(torch.stack([cross_entropy(output[b, :].reshape(1, -1), label.reshape(-1))
                                         for label in torch.where(multilabel_onehot_target[b, :] == 1)[0]], dim=0))
                                         for b in range(output.shape[0])]), dim=0)
    return multi_loss


# def multilabel_crossentropy_loss(output, current_cat, cat_padding_value):
#     # output (logits) - batch_size x cat_vocab_size
#     # current_cat - batch_size x max_cat_len
#     multi_loss = torch.sum(torch.stack([torch.sum(torch.stack([cross_entropy(output[b, :].reshape(1, -1), label.reshape(-1))
#                                          for label in current_cat[b, :torch.sum(current_cat[b, :]!=cat_padding_value).item()]], dim=0))
#                                          for b in range(output.shape[0])]), dim=0)
#     return multi_loss


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
    batch_size = 32
    dataloader_num_workers = 2

    optimizer_lr = 1e-4

    scheduler_factor = 0.3
    scheduler_patience = 5

    early_stopping_patience = 15

    model_name = 'Transformer1_time_corrected'
    results_folder = f'../results/{model_name}/'
    checkpoint = results_folder + f'checkpoints/look_back_{look_back}_pal.pt'

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    train_dataset = OrderReader(data_folder, train_file, test_file, valid_file, look_back, 'train')
    test_dataset = OrderReader(data_folder, train_file, test_file, valid_file, look_back, 'test')
    valid_dataset = OrderReader(data_folder, train_file, test_file, valid_file, look_back, 'valid')

    cat_vocab_size = train_dataset.cat_vocab_size
    id_vocab_size = train_dataset.id_vocab_size
    amount_vocab_size = train_dataset.amount_vocab_size
    dt_vocab_size = train_dataset.dt_vocab_size
    emb_dim = 128

    net = TransformerNet(look_back, cat_vocab_size, id_vocab_size, amount_vocab_size, dt_vocab_size, emb_dim).to(device)

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
            [batch_cat_arr, batch_current_cat, batch_dt_arr, batch_amount_arr, batch_id_arr] = batch_arrays
            optimizer.zero_grad()
            output_material = net(batch_cat_arr, batch_dt_arr, batch_amount_arr, batch_id_arr)

            batch_mask_current_cat = torch.tensor(~(batch_current_cat == cat_vocab_size),
                                                  dtype=torch.int64).unsqueeze(2).to(device)
            batch_onehot_current_cat = torch.sum(one_hot(batch_current_cat,
                                                         num_classes=cat_vocab_size+1) * batch_mask_current_cat, dim=1).to(device)

            loss = multilabel_crossentropy_loss(output_material, batch_onehot_current_cat)
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}/{num_epochs} || Train loss {epoch_train_loss}')

        print('Validation...')
        net.train(False)
        epoch_valid_loss = 0
        for batch_ind, batch_arrays in enumerate(valid_dataloader):
            batch_arrays = [arr.to(device) for arr in batch_arrays]
            [batch_cat_arr, batch_current_cat, batch_dt_arr, batch_amount_arr, batch_id_arr] = batch_arrays
            output_material = net(batch_cat_arr, batch_dt_arr, batch_amount_arr, batch_id_arr)

            batch_mask_current_cat = torch.tensor(~(batch_current_cat == cat_vocab_size),
                                                  dtype=torch.int64).unsqueeze(2).to(device)
            batch_onehot_current_cat = torch.sum(one_hot(batch_current_cat,
                                                         num_classes=cat_vocab_size+1) * batch_mask_current_cat, dim=1).to(device)

            loss = multilabel_crossentropy_loss(output_material, batch_onehot_current_cat)
            epoch_valid_loss += loss.item()

        print(f'Epoch {epoch}/{num_epochs} || Valid loss {epoch_valid_loss}')

        scheduler.step(epoch_valid_loss)

        early_stopping(epoch_valid_loss, net)
        if early_stopping.early_stop:
            print('Early stopping')
            break

# #----------------------------------------------
    net = TransformerNet(look_back, cat_vocab_size, id_vocab_size, amount_vocab_size, dt_vocab_size, emb_dim).to(device)

    net.load_state_dict(torch.load(checkpoint, map_location=device))
    net.train(False)
    print('Testing...')
    output_list = []
    gt_list = []
    for batch_ind, batch_arrays in enumerate(test_dataloader):
        batch_arrays = [arr.to(device) for arr in batch_arrays]
        [batch_cat_arr, batch_current_cat, batch_dt_arr, batch_amount_arr, batch_id_arr] = batch_arrays
        output_material = net(batch_cat_arr, batch_dt_arr, batch_amount_arr, batch_id_arr)

        batch_mask_current_cat = torch.tensor(~(batch_current_cat == cat_vocab_size),
                                              dtype=torch.int64).unsqueeze(2).to(device)
        batch_onehot_current_cat = torch.sum(one_hot(batch_current_cat,
                                                     num_classes=cat_vocab_size + 1) * batch_mask_current_cat, dim=1).to(device)

        output_list.append(output_material.detach().cpu())
        gt_list.append(batch_onehot_current_cat.detach().cpu())

    all_output = torch.cat(output_list, dim=0)
    all_gt = torch.cat(gt_list, dim=0)
    test_mean_patk = {i: mean_patk(all_output, all_gt, k=i) for i in range(1, 5)}
    test_mean_ratk = {i: mean_ratk(all_output, all_gt, k=i) for i in range(1, 5)}
    test_f1atk = {i: 2 * mean_patk(all_output, all_gt, k=i) * mean_ratk(all_output, all_gt, k=i) / (mean_patk(all_output, all_gt, k=i) + mean_ratk(all_output, all_gt, k=i))
                  for i in range(1, 5)}
    test_mapk = {i: mapk(all_output, all_gt, k=i) for i in range(1, 5)}

    print(f'Test Precision@k {test_mean_patk} || Test Recall@k {test_mean_ratk}|| Test F1@k {test_f1atk} || MAP@k {test_mapk})')

    with open(results_folder + f'{os.path.splitext(os.path.basename(checkpoint))[0]}.json', 'w', encoding='utf-8') as f:
        json.dump({'test_patk': test_mean_patk, 'test_ratk': test_mean_ratk, 'test_f1atk': test_f1atk, 'test_mapk': test_mapk}, f)

#-----------------------------------------------------------------------------

if __name__ == '__main__':
    train()
