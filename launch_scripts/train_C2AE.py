import torch
from torch import nn
from torch.utils.data import DataLoader
import sys
sys.path.append("../",)

from launch_scripts.train_lstm_material import mean_patk, mean_ratk, mapk
from models.regression.lstm_material import ClassificationNet
from models.C2AE.C2AE_class import C2AE, Fd, Fe, Fx
from data_preparation.data_reader_upd import OrderReader
import numpy as np
import os
import json
from tqdm import tqdm

torch.manual_seed(42)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train():

    data_folder = "/NOTEBOOK/UHH/Repository/Repository_LSTM/"
    train_file = "df_beer_train_nn.csv"
    test_file = "df_beer_test.csv"
    valid_file = "df_beer_valid_nn.csv"

    # data_folder = "/NOTEBOOK/UHH/"
    # train_file = "sales_train.csv"
    # test_file = "sales_test.csv"
    # valid_file = "sales_valid.csv"

    look_back = 3

    num_epochs = 100
    batch_size = 32
    dataloader_num_workers = 0

    optimizer_lr = 1e-6

    scheduler_factor = 0.9
    scheduler_patience = 5

    early_stopping_patience = 200
    model_name = 'LSTM_WITH_C2AE'
    results_folder = f'../C2AE_Sales_1exp_balancedLoss_lr1e-6/{model_name}/'
    checkpoint = results_folder + f'checkpoints/look_back_{look_back}_c2ae.pt'


    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("Use cuda")
    else:
        device = torch.device('cpu')
        print("Use cpu")

    train_dataset = OrderReader(data_folder, train_file, test_file, valid_file, look_back, 'train')
    test_dataset = OrderReader(data_folder, train_file, test_file, valid_file, look_back, 'test')
    valid_dataset = OrderReader(data_folder, train_file, test_file, valid_file, look_back, 'valid')

    linear_num_feat_dim = 32
    cat_embedding_dim = 512
    lstm_hidden_dim = 1024
    cat_vocab_size = train_dataset.cat_vocab_size
    id_vocab_size = train_dataset.id_vocab_size
    id_embedding_dim = 512
    linear_concat1_dim = 1024
    linear_concat2_dim = 512

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=dataloader_num_workers)

    onehot_current_cat_full = None
    for batch_ind, batch_arrays in enumerate(train_dataloader):
        batch_arrays = [arr.to(device) for arr in batch_arrays]
        [_, _, _, _, batch_onehot_current_cat, _, _, _] = batch_arrays
        if onehot_current_cat_full is None:
            onehot_current_cat_full = torch.sum(batch_onehot_current_cat, dim=0).float()
        else:
            onehot_current_cat_full += torch.sum(batch_onehot_current_cat, dim=0).float()
    onehot_current_cat_full /= torch.sum(onehot_current_cat_full)

    os.makedirs(results_folder+'checkpoints/', exist_ok=True)
    classifier_net = ClassificationNet(linear_num_feat_dim, cat_embedding_dim, lstm_hidden_dim,
                            cat_vocab_size, id_vocab_size,
                            id_embedding_dim, linear_concat1_dim, linear_concat2_dim)

    classifier_net.linear_material = nn.Linear(linear_concat2_dim, 512)
    classifier_net.bn3 = nn.BatchNorm1d(512)
    num_labels = cat_vocab_size + 1     # 61 + 1

    latent_dim = 512
    fx_hidden_dim = 1024
    fe_hidden_dim = 256
    fd_hidden_dim = 256

    fx = Fx(512, fx_hidden_dim, fx_hidden_dim, latent_dim).to(device)
    fe = Fe(num_labels, fe_hidden_dim, latent_dim).to(device)
    fd = Fd(latent_dim, fd_hidden_dim, num_labels, fin_act=torch.sigmoid).to(device)
    c2ae = C2AE(classifier_net.to(device), fx, fe, fd, beta=0.5, alpha=10, emb_lambda=0.01, latent_dim=latent_dim,
                device=device, onehot_current_cat_full=onehot_current_cat_full).to(device)

    #c2ae.load_state_dict(torch.load(checkpoint, map_location=device))
    #c2ae.train()

    optimizer = torch.optim.AdamW(c2ae.parameters(), lr=optimizer_lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor,
                                                           patience=scheduler_patience)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True, path=checkpoint)
    val_every_500 = False
    for epoch in range(1, num_epochs+1):
        c2ae.train()
        epoch_train_loss = 0
        print('Training...')
        for batch_ind, batch_arrays in enumerate(train_dataloader):

            # if batch_ind % 500 == 0:
            #     print(f'Validation step{batch_ind}.')
            #     c2ae.eval()
            #     output_list = []
            #     gt_list = []
            #     for batch_ind, batch_arrays in enumerate(valid_dataloader):
            #         batch_arrays = [arr.to(device) for arr in batch_arrays]
            #         [batch_cat_arr, batch_mask_cat,
            #          batch_current_cat, batch_mask_current_cat, batch_onehot_current_cat,
            #          batch_num_arr, batch_id_arr, batch_target] = batch_arrays  # current_minus1_cat] = batch_arrays
            #
            #         #batch_cat_arr_minus1 = torch.zeros((batch_cat_arr.shape[0],
            #         #                                    batch_cat_arr.shape[1]+1,
            #         #                                    batch_cat_arr.shape[2])).to(device)
            #         #batch_cat_arr_minus1[:, :-1] = batch_cat_arr
            #         #for i in range(round(batch_cat_arr.shape[2] * 0.7)):
            #         #    batch_cat_arr_minus1[:, -1, i] = batch_cat_arr[:, -1, 3]
            #         #batch_mask_cat_minus1 = torch.ones((batch_mask_cat.shape[0],
            #         #                                    batch_mask_cat.shape[1] + 1,
            #         #                                    batch_mask_cat.shape[2])).to(device)
            #         #batch_mask_cat_minus1[:, :-1] = batch_mask_cat
            #
            #         output = c2ae(batch_cat_arr, batch_mask_cat, batch_num_arr,
            #                       batch_id_arr)  # current_minus1_cat=current_minus1_cat)
            #         # output = c2ae(batch_cat_arr_minus1.long(), batch_mask_cat_minus1.long(), batch_num_arr, batch_id_arr)
            #         output_list.append(output.detach().cpu())
            #         gt_list.append(batch_onehot_current_cat.detach().cpu())
            #
            #     all_output = torch.cat(output_list, dim=0)
            #     all_gt = torch.cat(gt_list, dim=0)
            #     val_mean_patk = {i: mean_patk(all_output, all_gt, k=i) for i in range(1, 5)}
            #     val_mean_ratk = {i: mean_ratk(all_output, all_gt, k=i) for i in range(1, 5)}
            #     val_mapk = {i: mapk(all_output, all_gt, k=i) for i in range(1, 5)}
            #
            #     print(f'Valid Precision@k {val_mean_patk} || Valid Recall@k {val_mean_ratk}|| Valid MAP@k {val_mapk})')
            #
            #     scheduler.step(1 - 2 * ((np.mean(list(val_mean_patk.values())) *
            #                              np.mean(list(val_mean_ratk.values()))) /
            #                             (np.mean(list(val_mean_ratk.values())) +
            #                              np.mean(list(val_mean_patk.values())))))
            #     # scheduler.step()
            #     early_stopping(1 - 2 * ((np.mean(list(val_mean_patk.values())) *
            #                              np.mean(list(val_mean_ratk.values()))) /
            #                             (np.mean(list(val_mean_ratk.values())) +
            #                              np.mean(list(val_mean_patk.values())))), c2ae)
            #     if early_stopping.early_stop:
            #         print('Early stopping')
            #         break
            # c2ae.train()

            batch_arrays = [arr.to(device) for arr in batch_arrays]
            [batch_cat_arr, batch_mask_cat,
             batch_current_cat, batch_mask_current_cat, batch_onehot_current_cat,
             batch_num_arr, batch_id_arr, batch_target] = batch_arrays #current_minus1_cat] = batch_arrays

            optimizer.zero_grad()
            fx_x, fe_y, fd_z = c2ae(batch_cat_arr, batch_mask_cat, batch_num_arr, batch_id_arr,
                                    batch_onehot_current_cat) #, current_minus1_cat=current_minus1_cat)
            # Calc losses.
            l_loss, c_loss = c2ae.losses(fx_x, fe_y, fd_z, batch_onehot_current_cat)
            #gamma = 0.9
            loss = c2ae.beta * l_loss + c2ae.alpha * c_loss # + gamma * l_loss_t
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            #if val_every_500 and not ((batch_ind + 1) % 1000):
            #    break

        print(f'Epoch {epoch}/{num_epochs} || Train loss {epoch_train_loss}')

        print('Validation...')
        c2ae.eval()
        output_list = []
        gt_list = []
        for batch_ind, batch_arrays in enumerate(valid_dataloader):
            batch_arrays = [arr.to(device) for arr in batch_arrays]
            [batch_cat_arr, batch_mask_cat,
             batch_current_cat, batch_mask_current_cat, batch_onehot_current_cat,
             batch_num_arr, batch_id_arr, batch_target] = batch_arrays # current_minus1_cat] = batch_arrays

            # batch_cat_arr_minus1 = torch.zeros((batch_cat_arr.shape[0],
            #                                     batch_cat_arr.shape[1]+1,
            #                                     batch_cat_arr.shape[2])).to(device)
            # batch_cat_arr_minus1[:, :-1] = batch_cat_arr
            # for i in range(round(batch_cat_arr.shape[2] * 0.7)):
            #     batch_cat_arr_minus1[:, -1, i] = batch_cat_arr[:, -1, 3]
            # batch_mask_cat_minus1 = torch.ones((batch_mask_cat.shape[0],
            #                                     batch_mask_cat.shape[1] + 1,
            #                                     batch_mask_cat.shape[2])).to(device)
            # batch_mask_cat_minus1[:, :-1] = batch_mask_cat

            output = c2ae(batch_cat_arr, batch_mask_cat, batch_num_arr, batch_id_arr) #current_minus1_cat=current_minus1_cat)
            #output = c2ae(batch_cat_arr_minus1.long(), batch_mask_cat_minus1.long(), batch_num_arr, batch_id_arr)
            output_list.append(output.detach().cpu())
            gt_list.append(batch_onehot_current_cat.detach().cpu())

        all_output = torch.cat(output_list, dim=0)
        all_gt = torch.cat(gt_list, dim=0)
        val_mean_patk = {i: mean_patk(all_output, all_gt, k=i) for i in range(1, 5)}
        val_mean_ratk = {i: mean_ratk(all_output, all_gt, k=i) for i in range(1, 5)}
        val_mapk = {i: mapk(all_output, all_gt, k=i) for i in range(1, 5)}

        print(f'Valid Precision@k {val_mean_patk} || Valid Recall@k {val_mean_ratk}|| Valid MAP@k {val_mapk})')

        val_mean_patk = {i: mean_patk(all_output, all_gt, k=i) for i in range(1, 2)}
        val_mean_ratk = {i: mean_ratk(all_output, all_gt, k=i) for i in range(1, 2)}
        # if (1 - 2 * ((np.mean(list(val_mean_patk.values())) *
        #                   np.mean(list(val_mean_ratk.values()))) /
        #                (np.mean(list(val_mean_ratk.values())) +
        #                 np.mean(list(val_mean_patk.values()))))) < 0.29:
        #
        #     optimizer = torch.optim.AdamW(c2ae.parameters(), lr=optimizer_lr*0.001)
        #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.999,
        #                                                            patience=scheduler_patience)
        #     val_every_500 = True
        #     print("Fine tuning begin!")
        # else:
        #     scheduler.step(1 - 2 * ((np.mean(list(val_mean_patk.values())) *
        #                       np.mean(list(val_mean_ratk.values()))) /
        #                    (np.mean(list(val_mean_ratk.values())) +
        #                     np.mean(list(val_mean_patk.values())))))
        #scheduler.step()
        scheduler.step(1 - 2 * ((np.mean(list(val_mean_patk.values())) *
                               np.mean(list(val_mean_ratk.values()))) /
                            (np.mean(list(val_mean_ratk.values())) +
                             np.mean(list(val_mean_patk.values())))))
        early_stopping(1 - 2 * ((np.mean(list(val_mean_patk.values())) *
                          np.mean(list(val_mean_ratk.values()))) /
                       (np.mean(list(val_mean_ratk.values())) +
                        np.mean(list(val_mean_patk.values())))), c2ae)
        if early_stopping.early_stop:
            print('Early stopping')
            break

#------------------------------------------------------------

    #c2ae = C2AE(classifier_net, fx, fe, fd, latent_dim=latent_dim).to(device)
    c2ae.load_state_dict(torch.load(checkpoint, map_location=device))
    c2ae.eval()

    output_list = []
    gt_list = []
    for batch_ind, batch_arrays in enumerate(test_dataloader):
        batch_arrays = [arr.to(device) for arr in batch_arrays]
        [batch_cat_arr, batch_mask_cat,
         batch_current_cat, batch_mask_current_cat, batch_onehot_current_cat,
         batch_num_arr, batch_id_arr, batch_target] = batch_arrays

        # batch_cat_arr_minus1 = torch.zeros((batch_cat_arr.shape[0],
        #                                     batch_cat_arr.shape[1] + 1,
        #                                     batch_cat_arr.shape[2])).to(device)
        # batch_cat_arr_minus1[:, :-1] = batch_cat_arr
        # for i in range(round(batch_cat_arr.shape[2] * 0.7)):
        #     batch_cat_arr_minus1[:, -1, i] = batch_cat_arr[:, -1, 3]
        # batch_mask_cat_minus1 = torch.ones((batch_mask_cat.shape[0],
        #                                     batch_mask_cat.shape[1] + 1,
        #                                     batch_mask_cat.shape[2])).to(device)
        # batch_mask_cat_minus1[:, :-1] = batch_mask_cat

        #output = c2ae(batch_cat_arr_minus1.long(), batch_mask_cat_minus1.long(), batch_num_arr, batch_id_arr)

        output = c2ae(batch_cat_arr, batch_mask_cat, batch_num_arr, batch_id_arr) #current_minus1_cat=current_minus1_cat)
        output_list.append(output.detach().cpu())
        gt_list.append(batch_onehot_current_cat.detach().cpu())

    all_output = torch.cat(output_list, dim=0)
    all_gt = torch.cat(gt_list, dim=0)
    test_mean_patk = {i: mean_patk(all_output, all_gt, k=i) for i in range(1, 5)}
    test_mean_ratk = {i: mean_ratk(all_output, all_gt, k=i) for i in range(1, 5)}
    test_mapk = {i: mapk(all_output, all_gt, k=i) for i in range(1, 5)}

    print(f'Test Precision@k {test_mean_patk} || Test Recall@k {test_mean_ratk}|| Test MAP@k {test_mapk})')

    with open(results_folder + f'{os.path.splitext(os.path.basename(checkpoint))[0]}.json', 'w', encoding='utf-8') as f:
        json.dump({'test_precision@k': test_mean_patk, 'test_recall@k': test_mean_ratk, 'test_map@k': test_mapk}, f)

if __name__ == '__main__':
    train()
