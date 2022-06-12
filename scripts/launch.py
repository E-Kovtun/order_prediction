import torch
from torch.utils.data import DataLoader
from models.LANET import TransformerLabelNet
from data_preparation.data_reader import OrderReader
import os
from utils.earlystopping import EarlyStopping
import json
from utils.multilabel_loss import multilabel_crossentropy_loss
from tqdm import tqdm


def launch():
    with open('configs/base.json') as json_file:
        base_dict = json.load(json_file)
    prepared_folder, model_name, look_back, emb_dim, rand_seed_list = base_dict["prepared_folder"], base_dict["model_name"], \
        base_dict["look_back"], base_dict["emb_dim"], base_dict["rand_seed"]

    with open('configs/train_params.json') as json_file:
        train_params_dict = json.load(json_file)
    num_epochs, batch_size, dataloader_num_workers, optimizer_lr, scheduler_factor, scheduler_patience, early_stopping_patience = \
        train_params_dict["num_epochs"], train_params_dict["batch_size"], train_params_dict["dataloader_num_workers"], \
        train_params_dict["optimizer_lr"], train_params_dict["scheduler_factor"], train_params_dict["scheduler_patience"], \
        train_params_dict["early_stopping_patience"]

    for rand_seed in tqdm(rand_seed_list):
        torch.manual_seed(rand_seed)

        dataset_name = os.path.basename(os.path.normpath(prepared_folder))
        os.makedirs(os.path.join('checkpoints/', dataset_name, model_name), exist_ok=True)
        checkpoint = os.path.join('checkpoints/', dataset_name, model_name, f'checkpoint_look_back_{look_back}_seed_{rand_seed}.pt')

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        train_dataset = OrderReader(prepared_folder, look_back, 'train')
        valid_dataset = OrderReader(prepared_folder, look_back, 'valid')

        cat_vocab_size = train_dataset.cat_vocab_size
        id_vocab_size = train_dataset.id_vocab_size
        amount_vocab_size = train_dataset.amount_vocab_size
        dt_vocab_size = train_dataset.dt_vocab_size
        max_cat_len = train_dataset.max_cat_len

        net = TransformerLabelNet(look_back, cat_vocab_size, id_vocab_size, amount_vocab_size, dt_vocab_size, max_cat_len, emb_dim).to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=optimizer_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor,
                                                               patience=scheduler_patience)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_num_workers)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=dataloader_num_workers)

        early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True, path=checkpoint)

        for epoch in range(1, num_epochs+1):
            net.train(True)
            epoch_train_loss = 0
            print('Training...')
            for batch_ind, batch_arrays in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                batch_arrays = [arr.to(device) for arr in batch_arrays]
                [batch_cat_arr, batch_current_cat, batch_dt_arr, batch_amount_arr, batch_id_arr] = batch_arrays
                optimizer.zero_grad()
                conf_scores = net(batch_cat_arr, batch_dt_arr, batch_amount_arr, batch_id_arr)
                loss = multilabel_crossentropy_loss(conf_scores, batch_current_cat, cat_vocab_size)
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
                conf_scores = net(batch_cat_arr, batch_dt_arr, batch_amount_arr, batch_id_arr)
                loss = multilabel_crossentropy_loss(conf_scores, batch_current_cat, cat_vocab_size)
                epoch_valid_loss += loss.item()

            print(f'Epoch {epoch}/{num_epochs} || Valid loss {epoch_valid_loss}')

            scheduler.step(epoch_valid_loss)

            early_stopping(epoch_valid_loss, net)
            if early_stopping.early_stop:
                print('Early stopping')
                break


if __name__ == "__main__":
    launch()
