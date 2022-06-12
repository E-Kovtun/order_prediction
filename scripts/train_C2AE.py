import torch
from torch import nn
import sys
sys.path.append("../",)

from torch.utils.data import DataLoader
from data_preparation.data_reader import OrderReader
from utils.multilabel_metrics import calculate_all_metrics
from models.LSTM import LSTMnet
from models.C2AE import C2AE, Fd, Fe, Fx
#from data_preparation.data_reader_upd import OrderReader
import numpy as np
from torch.nn.functional import one_hot
import os
import json
from utils.multilabel_loss import multilabel_crossentropy_loss
from tqdm import tqdm
from utils.roc_auc_metrics import calculate_roc_auc_metrics
import pandas as pd

#torch.manual_seed(42)
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
    prepared_folder = "datasets/data/prepared/demand/"

    look_back = 3
    emb_dim = 128
    rand_seed_list = [1]
    num_epochs = 50
    batch_size = 32
    dataloader_num_workers = 0
    optimizer_lr = 1e-4
    scheduler_factor = 0.3
    scheduler_patience = 5
    early_stopping_patience = 15

    model_name = 'LSTM_WITH_C2AE'
    dataset_name = 'demand'
    all_roc_auc_micro = []
    all_roc_auc_macro = []
    for rand_seed in tqdm(rand_seed_list):
        torch.manual_seed(rand_seed)

        model_name = 'LSTM_WITH_C2AE'
        dataset_name = os.path.basename(os.path.normpath(prepared_folder))
        os.makedirs(os.path.join('checkpoints/', dataset_name, model_name), exist_ok=True)
        checkpoint = os.path.join('checkpoints/', dataset_name, model_name, f'checkpoint_look_back_{look_back}_seed_{rand_seed}.pt')

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        train_dataset = OrderReader(prepared_folder, look_back, 'train')
        valid_dataset = OrderReader(prepared_folder, look_back, 'valid')
        test_dataset = OrderReader(prepared_folder, look_back, 'test')
        # model_name = 'LSTM_WITH_C2AE'
        # results_folder = f'../C2AE_Sales_1exp_balancedLoss_lr1e-6/{model_name}/'
        # checkpoint = results_folder + f'checkpoints/look_back_{look_back}_c2ae.pt'


        # train_dataset = OrderReader(data_folder, train_file, test_file, valid_file, look_back, 'train')
        # test_dataset = OrderReader(data_folder, train_file, test_file, valid_file, look_back, 'test')
        # valid_dataset = OrderReader(data_folder, train_file, test_file, valid_file, look_back, 'valid')
        #

        linear_concat2_dim = 512

        cat_vocab_size = train_dataset.cat_vocab_size
        id_vocab_size = train_dataset.id_vocab_size
        amount_vocab_size = train_dataset.amount_vocab_size
        dt_vocab_size = train_dataset.dt_vocab_size
        max_cat_len = train_dataset.max_cat_len

        net = LSTMnet(look_back, cat_vocab_size, id_vocab_size, amount_vocab_size, dt_vocab_size,
                                  max_cat_len, emb_dim).to(device)

        # classifier_net = ClassificationNet(linear_num_feat_dim, cat_embedding_dim, lstm_hidden_dim,
        #                         cat_vocab_size, id_vocab_size,
        #                         id_embedding_dim, linear_concat1_dim, linear_concat2_dim)

        net.linear_history2 = nn.Linear(2 * emb_dim, 512)
        #net.linear_material = nn.Linear(linear_concat2_dim, 512)
        #net.bn3 = nn.BatchNorm1d(512)
        num_labels = cat_vocab_size + 1     # 61 + 1

        latent_dim = 512
        fx_hidden_dim = 1024
        fe_hidden_dim = 256
        fd_hidden_dim = 256

        fx = Fx(512, fx_hidden_dim, fx_hidden_dim, latent_dim).to(device)
        fe = Fe(num_labels, fe_hidden_dim, latent_dim).to(device)
        fd = Fd(latent_dim, fd_hidden_dim, num_labels, fin_act=torch.sigmoid).to(device)
        c2ae = C2AE(net.to(device), fx, fe, fd, beta=0.5, alpha=10, emb_lambda=0.01, latent_dim=latent_dim,
                    device=device).to(device)
        optimizer = torch.optim.Adam(c2ae.parameters(), lr=optimizer_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor,
                                                               patience=scheduler_patience)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=dataloader_num_workers)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=dataloader_num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=dataloader_num_workers)

        early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True, path=checkpoint)

        for epoch in range(1, num_epochs+1):
            c2ae.train(True)
            epoch_train_loss = 0
            print('Training...')
            for batch_ind, batch_arrays in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                batch_arrays = [arr.to(device) for arr in batch_arrays]
                [batch_cat_arr, batch_current_cat, batch_dt_arr, batch_amount_arr, batch_id_arr, current_minus1_cat] = batch_arrays
                optimizer.zero_grad()
                batch_mask_current_cat = torch.tensor(~(batch_current_cat == cat_vocab_size),
                                                      dtype=torch.int64).unsqueeze(2).to(device)
                batch_onehot_current_cat = torch.sum(one_hot(batch_current_cat,
                                                             num_classes=cat_vocab_size + 1) * batch_mask_current_cat,
                                                     dim=1).to(device)
                fx_x, fe_y, fd_z, fe_y_t = c2ae(batch_cat_arr, batch_dt_arr, batch_amount_arr, batch_id_arr,
                                        batch_onehot_current_cat=batch_onehot_current_cat, current_minus1_cat=current_minus1_cat)
                # Calc losses.
                l_loss, c_loss, l_loss_t = c2ae.losses(fx_x, fe_y, fd_z, batch_onehot_current_cat, fe_y_t)
                gamma = 0.2
                loss = c2ae.beta * l_loss + c2ae.alpha * c_loss + gamma * l_loss_t
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()


            print(f'Epoch {epoch}/{num_epochs} || Train loss {epoch_train_loss}')

            print('Validation...')

            c2ae.train(False)
            epoch_valid_loss = 0
            for batch_ind, batch_arrays in enumerate(valid_dataloader):
                batch_arrays = [arr.to(device) for arr in batch_arrays]
                [batch_cat_arr, batch_current_cat, batch_dt_arr, batch_amount_arr, batch_id_arr, current_minus1_cat] = batch_arrays # current_minus1_cat] = batch_arrays

                conf_scores = c2ae(batch_cat_arr, batch_dt_arr, batch_amount_arr, batch_id_arr) #current_minus1_cat=current_minus1_cat)

                loss = multilabel_crossentropy_loss(conf_scores, batch_current_cat, cat_vocab_size)
                epoch_valid_loss += loss.item()
                #For to provide some metrics
                # batch_mask_current_cat = torch.tensor(~(batch_current_cat == cat_vocab_size),
                #                                       dtype=torch.int64).unsqueeze(
                #     2).to(device)
                # batch_onehot_current_cat = torch.sum(one_hot(batch_current_cat,
                #                                              num_classes=cat_vocab_size + 1) * batch_mask_current_cat,
                #                                      dim=1).to(device)
                #
                # gt_valid.extend(batch_onehot_current_cat[:, :-1].detach().cpu().tolist())
                # scores_valid.extend(conf_scores.tolist())

            print(f'Epoch {epoch}/{num_epochs} || Valid loss {epoch_valid_loss}')

            scheduler.step(epoch_valid_loss)

            early_stopping(epoch_valid_loss, c2ae)
            if early_stopping.early_stop:
                print('Early stopping')
                break
    #------------------------------------------------------------

        #c2ae = C2AE(classifier_net, fx, fe, fd, latent_dim=latent_dim).to(device)
        c2ae.load_state_dict(torch.load(checkpoint, map_location=device))
        c2ae.eval()
        print('Testing...')
        all_scores = []
        all_gt = []
        for batch_ind, batch_arrays in enumerate(test_dataloader):
            batch_arrays = [arr.to(device) for arr in batch_arrays]
            [batch_cat_arr, batch_current_cat, batch_dt_arr, batch_amount_arr, batch_id_arr, current_minus1_cat] = batch_arrays

            conf_scores = c2ae(batch_cat_arr, batch_dt_arr, batch_amount_arr, batch_id_arr).detach().cpu()

            batch_mask_current_cat = torch.tensor(~(batch_current_cat == cat_vocab_size), dtype=torch.int64).unsqueeze(
                2).to(device)
            batch_onehot_current_cat = torch.sum(one_hot(batch_current_cat,
                                                         num_classes=cat_vocab_size + 1) * batch_mask_current_cat,
                                                 dim=1).to(device)

            all_gt.extend(batch_onehot_current_cat[:, :-1].detach().cpu().tolist())
            all_scores.extend(conf_scores.tolist())

        metrics_dict = calculate_roc_auc_metrics(np.array(all_scores), np.array(all_gt))
        print(metrics_dict)
        all_roc_auc_micro.append(metrics_dict['roc_auc_micro'])
        all_roc_auc_macro.append(metrics_dict['roc_auc_macro'])
        os.makedirs(os.path.join('results/', dataset_name, model_name), exist_ok=True)
        with open(os.path.join('results/', dataset_name, model_name,
                               f'metrics_look_back_{look_back}_seed_{rand_seed}.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f)

    df = pd.DataFrame(data=[[np.mean(all_roc_auc_micro), np.std(all_roc_auc_micro)],
                            [np.mean(all_roc_auc_macro), np.std(all_roc_auc_macro)]],
                      index=['roc_auc_micro', 'roc_auc_macro'],
                      columns=['mean', 'std'])
    df.to_csv(os.path.join('results/', dataset_name, model_name, f'metrics_final.csv'))

        # For only testing
        # metrics_dict = calculate_all_metrics(np.array(all_gt), np.array(all_scores),
        #                                      np.array(gt_valid), np.array(scores_valid), kind='thr')
        # test_mean_patk = {i: mean_patk(torch.tensor(all_scores), torch.tensor(all_gt), k=i) for i in range(1, 5)}
        # test_mean_ratk = {i: mean_ratk(torch.tensor(all_scores), torch.tensor(all_gt), k=i) for i in range(1, 5)}
        # test_mapk = {i: mapk(torch.tensor(all_scores), torch.tensor(all_gt), k=i) for i in range(1, 5)}
        #
        # print(f'Test Precision@k {test_mean_patk} || Test Recall@k {test_mean_ratk}|| Test MAP@k {test_mapk})')






if __name__ == "__main__":
    train()