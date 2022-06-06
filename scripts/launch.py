import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
from models.LSTM import LSTMnet
from models.TransformerLabel import TransformerLabelNet
from data_preparation.data_reader import OrderReader
import numpy as np
import os
from utils.earlystopping import EarlyStopping
import json
from utils.multilabel_loss import multilabel_crossentropy_loss
from utils.multilabel_metrics import calculate_all_metrics
from tqdm import tqdm


def launch():
    with open('configs/base.json') as json_file:
        base_dict = json.load(json_file)
    prepared_folder, model_name, look_back, emb_dim, rand_seed = base_dict["prepared_folder"], base_dict["model_name"], \
        base_dict["look_back"], base_dict["emb_dim"], base_dict["rand_seed"]

    with open('configs/train_params.json') as json_file:
        train_params_dict = json.load(json_file)
    num_epochs, batch_size, dataloader_num_workers, optimizer_lr, scheduler_factor, scheduler_patience, early_stopping_patience = \
        train_params_dict["num_epochs"], train_params_dict["batch_size"], train_params_dict["dataloader_num_workers"], \
        train_params_dict["optimizer_lr"], train_params_dict["scheduler_factor"], train_params_dict["scheduler_patience"], \
        train_params_dict["early_stopping_patience"]

    torch.manual_seed(rand_seed)

    dataset_name = os.path.basename(os.path.normpath(prepared_folder))
    os.makedirs(os.path.join('checkpoints/', dataset_name, model_name), exist_ok=True)
    checkpoint = os.path.join('checkpoints/', dataset_name, model_name, 'checkpoint.pt')

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    train_dataset = OrderReader(prepared_folder, look_back, 'train')
    test_dataset = OrderReader(prepared_folder, look_back, 'test')
    valid_dataset = OrderReader(prepared_folder, look_back, 'valid')

    cat_vocab_size = train_dataset.cat_vocab_size
    id_vocab_size = train_dataset.id_vocab_size
    amount_vocab_size = train_dataset.amount_vocab_size
    dt_vocab_size = train_dataset.dt_vocab_size
    max_cat_len = train_dataset.max_cat_len

    if model_name == "LSTMnet":
        net = LSTMnet(look_back, cat_vocab_size, id_vocab_size, amount_vocab_size, dt_vocab_size, max_cat_len, emb_dim).to(device)
    elif model_name == "TransformerLabel":
        net = TransformerLabelNet(look_back, cat_vocab_size, id_vocab_size, amount_vocab_size, dt_vocab_size, max_cat_len, emb_dim).to(device)
    else:
        print("Model is not implemented")

    optimizer = torch.optim.Adam(net.parameters(), lr=optimizer_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor,
                                                           patience=scheduler_patience)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=dataloader_num_workers)

    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True, path=checkpoint)
    num_loss = nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs+1):
        net.train(True)
        epoch_train_loss = 0
        print('Training...')
        for batch_ind, batch_arrays in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            batch_arrays = [arr.to(device) for arr in batch_arrays]
            [batch_cat_arr, batch_current_cat, batch_dt_arr, batch_amount_arr, batch_id_arr, batch_number_arr] = batch_arrays
            optimizer.zero_grad()
            conf_scores, output_num = net(batch_cat_arr, batch_dt_arr, batch_amount_arr, batch_id_arr)
            loss = 0.01 * multilabel_crossentropy_loss(conf_scores, batch_current_cat, cat_vocab_size) + num_loss(output_num, batch_number_arr)
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}/{num_epochs} || Train loss {epoch_train_loss}')

        print('Validation...')
        net.train(False)
        epoch_valid_loss = 0
        for batch_ind, batch_arrays in enumerate(valid_dataloader):
            batch_arrays = [arr.to(device) for arr in batch_arrays]
            [batch_cat_arr, batch_current_cat, batch_dt_arr, batch_amount_arr, batch_id_arr, batch_number_arr] = batch_arrays
            conf_scores, output_num = net(batch_cat_arr, batch_dt_arr, batch_amount_arr, batch_id_arr)
            loss = 0.01 * multilabel_crossentropy_loss(conf_scores, batch_current_cat, cat_vocab_size) + num_loss(output_num, batch_number_arr)
            epoch_valid_loss += loss.item()

        print(f'Epoch {epoch}/{num_epochs} || Valid loss {epoch_valid_loss}')

        scheduler.step(epoch_valid_loss)

        early_stopping(epoch_valid_loss, net)
        if early_stopping.early_stop:
            print('Early stopping')
            break

#------------------------------------------------------

    net.load_state_dict(torch.load(checkpoint, map_location=device))
    net.train(False)
    print('Testing...')
    all_preds = []
    all_scores = []
    all_gt = []
    for batch_ind, batch_arrays in enumerate(test_dataloader):
        batch_arrays = [arr.to(device) for arr in batch_arrays]
        [batch_cat_arr, batch_current_cat, batch_dt_arr, batch_amount_arr, batch_id_arr, batch_number_arr] = batch_arrays

        conf_scores, output_num = net(batch_cat_arr, batch_dt_arr, batch_amount_arr, batch_id_arr)

        batch_mask_current_cat = torch.tensor(~(batch_current_cat == cat_vocab_size), dtype=torch.int64).unsqueeze(2).to(device)
        batch_onehot_current_cat = torch.sum(one_hot(batch_current_cat,
                                                     num_classes=cat_vocab_size+1) * batch_mask_current_cat, dim=1).to(device)

        pred = [torch.zeros(cat_vocab_size, dtype=torch.int64).index_fill_(dim=0,
               index=torch.topk(conf_scores[b, :], dim=0, k=torch.argmax(output_num[b, :], dim=0) + 1).indices,
               value=1).tolist() for b in range(conf_scores.shape[0])]
        all_preds.extend(pred)
        all_gt.extend(batch_onehot_current_cat[:, :-1].detach().cpu().tolist())
        all_scores.extend(conf_scores.detach().cpu().numpy())

    metrics_dict = calculate_all_metrics(np.array(all_preds), np.array(all_gt), np.array(all_scores))
    os.makedirs(os.path.join('results/', dataset_name, model_name), exist_ok=True)
    with open(os.path.join('results/', dataset_name, model_name, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f)


if __name__ == "__main__":
    launch()
