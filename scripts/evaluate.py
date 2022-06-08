from models.LANET import TransformerLabelNet
from data_preparation.data_reader import OrderReader
from utils.roc_auc_metrics import calculate_roc_auc_metrics
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
import json
import torch
import os
from tqdm import tqdm
import numpy as np
import pandas as pd


def evaluate():
    with open('configs/base.json') as json_file:
        base_dict = json.load(json_file)
    prepared_folder, model_name, look_back, emb_dim, rand_seed_list = base_dict["prepared_folder"], base_dict["model_name"], \
        base_dict["look_back"], base_dict["emb_dim"], base_dict["rand_seed"]

    all_roc_auc_micro = []
    all_roc_auc_macro = []
    for rand_seed in tqdm(rand_seed_list):
        torch.manual_seed(rand_seed)

        dataset_name = os.path.basename(os.path.normpath(prepared_folder))
        checkpoint = os.path.join('checkpoints/', dataset_name, model_name, f'checkpoint_look_back_{look_back}_seed_{rand_seed}.pt')

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        test_dataset = OrderReader(prepared_folder, look_back, 'test')
        cat_vocab_size = test_dataset.cat_vocab_size
        id_vocab_size = test_dataset.id_vocab_size
        amount_vocab_size = test_dataset.amount_vocab_size
        dt_vocab_size = test_dataset.dt_vocab_size
        max_cat_len = test_dataset.max_cat_len

        test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=2)
        net = TransformerLabelNet(look_back, cat_vocab_size, id_vocab_size, amount_vocab_size, dt_vocab_size, max_cat_len, emb_dim).to(device)
        net.load_state_dict(torch.load(checkpoint, map_location=device))
        net.train(False)
        print('Testing...')
        all_scores = []
        all_gt = []
        for batch_ind, batch_arrays in enumerate(test_dataloader):
            batch_arrays = [arr.to(device) for arr in batch_arrays]
            [batch_cat_arr, batch_current_cat, batch_dt_arr, batch_amount_arr, batch_id_arr] = batch_arrays

            conf_scores = net(batch_cat_arr, batch_dt_arr, batch_amount_arr, batch_id_arr).detach().cpu()

            batch_mask_current_cat = torch.tensor(~(batch_current_cat == cat_vocab_size), dtype=torch.int64).unsqueeze(2).to(device)
            batch_onehot_current_cat = torch.sum(one_hot(batch_current_cat,
                                                         num_classes=cat_vocab_size+1) * batch_mask_current_cat, dim=1).to(device)

            all_gt.extend(batch_onehot_current_cat[:, :-1].detach().cpu().tolist())
            all_scores.extend(conf_scores.tolist())

        metrics_dict = calculate_roc_auc_metrics(np.array(all_scores), np.array(all_gt))
        all_roc_auc_micro.append(metrics_dict['roc_auc_micro'])
        all_roc_auc_macro.append(metrics_dict['roc_auc_macro'])
        os.makedirs(os.path.join('results/', dataset_name, model_name), exist_ok=True)
        with open(os.path.join('results/', dataset_name, model_name, f'metrics_look_back_{look_back}_seed_{rand_seed}.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f)

    df = pd.DataFrame(data=[[np.mean(all_roc_auc_micro), np.std(all_roc_auc_micro)], [np.mean(all_roc_auc_macro), np.std(all_roc_auc_macro)]],
                      index=['roc_auc_micro', 'roc_auc_macro'],
                      columns=['mean', 'std'])
    df.to_csv(os.path.join('results/', dataset_name, model_name, f'metrics_final.csv'))

if __name__ == "__main__":
    evaluate()
