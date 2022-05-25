import torch
from torch import nn
from torch.utils.data import DataLoader
import sys
sys.path.append("../",)
#from models.regression.number_material import NumberNet
from models.regression.lstm_material import ClassificationNet
from models.C2AE.C2AE_class import C2AE, Fd, Fe, Fx
from data_preparation.data_reader_transactions import OrderReader
from torch.nn.functional import one_hot
import numpy as np
import os
import json
from tqdm import tqdm
from utils.earlystopping import EarlyStopping
torch.manual_seed(2)
from utils.utils import get_max_cat_len
import warnings
warnings.filterwarnings("ignore")


class NumberNet(nn.Module):
    def __init__(self, cat_vocab_size, max_cat_len):
        super(NumberNet, self).__init__()
        self.max_cat_len = max_cat_len

        self.linear1 = nn.Linear(cat_vocab_size + 1, 2 * max_cat_len)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2 * max_cat_len, max_cat_len)

    def forward(self, x_material):
        x1 = self.linear1(x_material)
        x1 = self.relu(x1)
        x2 = self.linear2(x1)
        return x2

def precision_mat(pred_mat, gt_mat):
    return len(np.intersect1d(pred_mat, gt_mat)) / len(pred_mat) if len(pred_mat) > 0 else 0


def recall_mat(pred_mat, gt_mat):
    return len(np.intersect1d(pred_mat, gt_mat)) / len(gt_mat)


def f1_mat(pred_mat, gt_mat):
    return 2 * precision_mat(pred_mat, gt_mat) * recall_mat(pred_mat, gt_mat) / (precision_mat(pred_mat, gt_mat) + recall_mat(pred_mat, gt_mat)) \
           if (precision_mat(pred_mat, gt_mat) + recall_mat(pred_mat, gt_mat)) > 0 else 0


def train():
    data_folder = "/NOTEBOOK/UHH/Repository/Repository_LSTM/"
    train_file = "df_beer_train_nn.csv"
    test_file = "df_beer_test.csv"
    valid_file = "df_beer_valid_nn.csv"

    # train_file = "sales_train.csv"
    # test_file = "sales_test.csv"
    # valid_file = "sales_valid.csv"

    look_back = 3

    num_epochs = 10
    batch_size = 32
    dataloader_num_workers = 0

    optimizer_lr = 1e-4

    scheduler_factor = 0.3
    scheduler_patience = 5

    early_stopping_patience = 15
    model_name = 'LSTM_WITH_C2AE'
    results_folder = f'../firstexp_a_10_ver2+bn_gpu_100xhidden_10xlatent/{model_name}/'
    checkpoint = results_folder + f'checkpoints/look_back_{look_back}_c2ae.pt'
    checkpoint_num = results_folder + f'checkpoints/look_back_{look_back}_Cross_Entropy_num.pt'

    if not torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    train_dataset = OrderReader(data_folder, train_file, test_file, valid_file, look_back, 'train')
    test_dataset = OrderReader(data_folder, train_file, test_file, valid_file, look_back, 'test')
    valid_dataset = OrderReader(data_folder, train_file, test_file, valid_file, look_back, 'valid')

    linear_num_feat_dim = 32
    cat_embedding_dim = 512
    lstm_hidden_dim = 1024
    id_embedding_dim = 512
    linear_concat1_dim = 1024
    linear_concat2_dim = 512
    cat_vocab_size = train_dataset.cat_vocab_size
    id_vocab_size = train_dataset.id_vocab_size
    max_cat_len = train_dataset.max_cat_len

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=dataloader_num_workers)

    os.makedirs(results_folder+'checkpoints/', exist_ok=True)

    number_net = NumberNet(cat_vocab_size, max_cat_len).to(device)
    classifier_net = ClassificationNet(linear_num_feat_dim, cat_embedding_dim, lstm_hidden_dim,
                                       cat_vocab_size, id_vocab_size,
                                       id_embedding_dim, linear_concat1_dim, linear_concat2_dim)

    classifier_net.linear_material = nn.Linear(linear_concat2_dim, 512)
    classifier_net.bn3 = nn.BatchNorm1d(512)
    num_labels = cat_vocab_size + 1  # 61 + 1

    latent_dim = 512
    fx_hidden_dim = 1024
    fe_hidden_dim = 256
    fd_hidden_dim = 256

    fx = Fx(512, fx_hidden_dim, fx_hidden_dim, latent_dim).to(device)
    fe = Fe(num_labels, fe_hidden_dim, latent_dim).to(device)
    fd = Fd(latent_dim, fd_hidden_dim, num_labels, fin_act=torch.sigmoid).to(device)

    net = C2AE(classifier_net.to(device), fx, fe, fd, beta=0.5, alpha=10, emb_lambda=0.01, latent_dim=latent_dim,
                device=device).to(device)
    net.load_state_dict(torch.load(checkpoint, map_location=device))
    net.classification.train(False)
    net.train(False)
    classification_loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(number_net.parameters(), lr=optimizer_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor,
                                                           patience=scheduler_patience)
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True, path=checkpoint_num)

    for epoch in range(1, num_epochs + 1):
        number_net.train(True)
        epoch_train_loss = 0
        print('Training...')
        for batch_ind, batch_arrays in enumerate(train_dataloader):
            batch_arrays = [arr.to(device) for arr in batch_arrays]
            [batch_cat_arr, batch_mask_cat,
             batch_current_cat, batch_mask_current_cat, batch_onehot_current_cat,
             batch_num_arr, batch_id_arr, batch_target] = batch_arrays
            optimizer.zero_grad()
            output_material = net(batch_cat_arr, batch_mask_cat, batch_num_arr, batch_id_arr, batch_onehot_current_cat)
            output_num = number_net(output_material)
            batch_gt_labels = batch_target
            loss = classification_loss(output_num, batch_gt_labels) #.float().reshape(-1, 1))
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}/{num_epochs} || Train loss {epoch_train_loss}')

        print('Validation...')
        number_net.train(False)
        epoch_valid_loss = 0
        for batch_ind, batch_arrays in enumerate(valid_dataloader):
            batch_arrays = [arr.to(device) for arr in batch_arrays]
            [batch_cat_arr, batch_mask_cat,
             batch_current_cat, batch_mask_current_cat, batch_onehot_current_cat,
             batch_num_arr, batch_id_arr, batch_target] = batch_arrays
            output_material = net(batch_cat_arr, batch_mask_cat, batch_num_arr, batch_id_arr)
            output_num = number_net(output_material)
            batch_gt_labels = batch_target
            loss = classification_loss(output_num, batch_gt_labels)
            epoch_valid_loss += loss.item()

        print(f'Epoch {epoch}/{num_epochs} || Valid loss {epoch_valid_loss}')

        scheduler.step(epoch_valid_loss)

        early_stopping(epoch_valid_loss, number_net)
        if early_stopping.early_stop:
            print('Early stopping')
            break

    # ------------------------------------------------------------

    number_net = NumberNet(cat_vocab_size, 13).to(device)
    number_net.load_state_dict(torch.load(checkpoint_num, map_location=device))
    number_net.train(False)

    all_precision = []
    all_recall = []
    all_f1 = []
    for batch_ind, batch_arrays in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        batch_arrays = [arr.to(device) for arr in batch_arrays]
        [batch_cat_arr, batch_mask_cat,
         batch_current_cat, batch_mask_current_cat, batch_onehot_current_cat,
         batch_num_arr, batch_id_arr, batch_target] = batch_arrays
        output_material = net(batch_cat_arr, batch_mask_cat, batch_num_arr, batch_id_arr)
        output_num = number_net(output_material)

        predicted_materials = [
            torch.topk(output_material[b, :], dim=0, k=torch.argmax(output_num[b, :], dim=0) + 1).indices.tolist()
            for b in range(output_material.shape[0])]
        gt_materials = [np.where(batch_onehot_current_cat.detach().cpu().numpy()[b, :] == 1)[0].tolist()
                        for b in range(output_material.shape[0])]

        batch_precision = [precision_mat(predicted_materials[b], gt_materials[b]) for b in
                           range(output_material.shape[0])]
        batch_recall = [recall_mat(predicted_materials[b], gt_materials[b]) for b in range(output_material.shape[0])]
        batch_f1 = [f1_mat(predicted_materials[b], gt_materials[b]) for b in range(output_material.shape[0])]

        all_precision.extend(batch_precision)
        all_recall.extend(batch_recall)
        all_f1.extend(batch_f1)

    test_precision = np.mean(all_precision)
    test_recall = np.mean(all_recall)
    test_f1 = np.mean(all_f1)

    print(f'Test Precision || {test_precision} || Test Recall {test_recall} || Test F1 {test_f1}')

    with open(results_folder + f'{os.path.splitext(os.path.basename(checkpoint_num))[0]}.json', 'w',
              encoding='utf-8') as f:
        json.dump({'test_precision': test_precision, 'test_recall': test_recall, 'test_f1': test_f1}, f)


if __name__ == '__main__':
    train()