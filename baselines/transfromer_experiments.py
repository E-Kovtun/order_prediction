import sys
# sys.path.append('/data/zabolotny-av/jupyter_root/src/order_prediction/order_prediction-general_development/launch_scripts/datasets/SRC_NEW')


from catalyst import dl
import catalyst


import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy, one_hot
import numpy as np
import os
from utils.earlystopping import EarlyStopping
import json
from tqdm import tqdm
from tqdm.notebook import tqdm

import pandas as pd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device')
parser.add_argument('--seed')
parser.add_argument('--dataset')

args = parser.parse_args()
args = vars(args)
print(args)

DATASET = str(args['dataset'])
device= str(args['device'])
seed = int(args['seed'])


catalyst.utils.set_global_seed(seed)

EXP_NAME = f'base_seed={seed}_device={device}'


from data_preparation.dataset_preparation import OrderDataset
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import torch
from torch.nn.functional import pad

class OrderReader(Dataset):
    def __init__(self, data_folder, look_back, phase):
        super(OrderReader, self).__init__()
        self.order_dataset = OrderDataset(data_folder, look_back)
        self.phase = phase

        [train_final, test_final, valid_final], cat_vocab_size, id_vocab_size, amount_vocab_size, dt_vocab_size,             max_cat_len = self.order_dataset.preprocess_dataframe()

        if self.phase == 'train':
            self.df_final, self.ind_combinations = train_final, self.order_dataset.window_combinations(train_final)
        if self.phase == 'test':
            self.df_final, self.ind_combinations = test_final, self.order_dataset.window_combinations(test_final)
        if self.phase == 'valid':
            self.df_final, self.ind_combinations = valid_final, self.order_dataset.window_combinations(valid_final)

        self.cat_vocab_size = cat_vocab_size
        self.id_vocab_size = id_vocab_size
        self.amount_vocab_size = amount_vocab_size
        self.dt_vocab_size = dt_vocab_size

        self.max_cat_len = max_cat_len

    def __len__(self):
        return len(self.ind_combinations)
    
    def _process_row_cat(self, row, ):
        MAX_CAT = self.cat_vocab_size + 1
        num_tensor =torch.zeros((1,MAX_CAT)).float()
        for cat, amount in zip(
            row[self.order_dataset.categorical],
            row[self.order_dataset.amount]
        ):        
            num_tensor[0, cat] = amount + 1

        dt_tensor = torch.tensor(row[self.order_dataset.dt]).reshape(1,1)
        id_tensor = torch.tensor(row[self.order_dataset.id]).reshape(1,1)
        cat_tensor = torch.hstack((dt_tensor, id_tensor))

        return num_tensor, cat_tensor

    def __getitem__(self, index):        
        input_indexes = self.ind_combinations[index][0]
        target_index = self.ind_combinations[index][1]
        
        num_tensor, cat_tensor = [],[]
        for idx in input_indexes:
            num_row, cat_row = self._process_row_cat(self.df_final.iloc[idx])
            num_tensor.append(num_row), cat_tensor.append(cat_row)        
            
        num_tensor, cat_tensor = torch.vstack(num_tensor), torch.vstack(cat_tensor)        

        target_row, _ = self._process_row_cat(self.df_final.iloc[target_index])
        targets = (target_row > 0.).float()[0][:,None]

        return {
            'categorical' : cat_tensor,
            'numerical' : num_tensor,
            'targets' : targets
        }


def multilabel_crossentropy_loss(output, multilabel_onehot_target):
    # output (logits) - batch_size x cat_vocab_size
    # multilabel_onehot_target - batch_size x cat_vocab_size + 1
    multi_loss = torch.sum(torch.stack([torch.sum(torch.stack([cross_entropy(output[b, :].reshape(1, -1), label.reshape(-1))
                                         for label in torch.where(multilabel_onehot_target[b, :] == 1)[0]], dim=0))
                                         for b in range(output.shape[0])]), dim=0)
    return multi_loss


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


data_folder = os.path.join("/data/zabolotny-av/sequential_data_paper/", DATASET)
look_back = 3

num_epochs = 500
batch_size = 32
dataloader_num_workers = 2

optimizer_lr = 1e-4
alpha = 1e-2

scheduler_factor = 0.3
scheduler_patience = 5

early_stopping_patience = 15

if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')

print('train')
train_dataset = OrderReader(data_folder, look_back, 'train')

print('test')
test_dataset = OrderReader(data_folder, look_back, 'test')
print('valid')
valid_dataset = OrderReader(data_folder, look_back, 'valid')


cat_vocab_size = train_dataset.cat_vocab_size
id_vocab_size = train_dataset.id_vocab_size
amount_vocab_size = train_dataset.amount_vocab_size
dt_vocab_size = train_dataset.dt_vocab_size
emb_dim = 128


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_num_workers)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=dataloader_num_workers)


class Baseline(torch.nn.Module):
    def __init__(self, data_shapes={'numerical' : [], 'categorical' : []}, model_params={
        'emb_size' : 32, 'num_out_size':64,'classes' : cat_vocab_size + 1, 'transformer_layers' : 2
    }):
        super().__init__()
        self.num_linear = torch.nn.Linear(
            data_shapes['numerical'],
            model_params['emb_size']
        )
        self.emb_layers = torch.nn.ModuleList(
            torch.nn.Embedding(
                number_of_categories, 
                model_params['emb_size']
            ) for number_of_categories in data_shapes['categorical']
        )
        self.act = torch.nn.ReLU()
        self.total_cat_emb_size = len(data_shapes['categorical']) * model_params['emb_size']
        self.act = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(data_shapes['numerical'])
        self.numerical = torch.nn.Sequential(
            
            torch.nn.Linear(data_shapes['numerical'], 64),
            self.act,
            torch.nn.Linear(64, model_params['num_out_size']),
        )
        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(model_params['num_out_size'] * 2, model_params['num_out_size']),
            self.act, 
            torch.nn.Linear(model_params['num_out_size'], model_params['classes'])
        )
                    
        self.transformer = torch.nn.TransformerEncoder(
        torch.nn.TransformerEncoderLayer(
                model_params['num_out_size'] + self.total_cat_emb_size,
                4, dim_feedforward=(model_params['num_out_size'] + self.total_cat_emb_size) * 4,
            ),model_params['transformer_layers'])
            
        
        
    def _forward_numerical(self, numerical):
        x = self.bn(numerical.transpose(1,2)).transpose(1,2)
        return self.numerical(x)
                
    def _forward_categorical(self, categorical):
        out = []
        for idx, emb_layer  in enumerate(self.emb_layers):
            out.append(
                emb_layer(
                    categorical[..., idx]
                )
            )
        return torch.cat(out, axis=-1)
    
    def forward(self, numerical, categorical):
        cat_embedding = self._forward_categorical(categorical)        
        numerical = self._forward_numerical(numerical)        
        obj_embeddings = torch.cat((cat_embedding, numerical), axis=-1)                        
        tr_out = self.transformer(obj_embeddings.transpose(0,1)).transpose(0,1)
        
        tr_out_mean = tr_out.mean(axis=1)        
        
        logits = self.cls_head(tr_out_mean)[..., None]
        return {'logits' : logits}

base_log_path = os.path.join('/data/zabolotny-av/logs/order_prediction/datasets', DATASET)
exp_log_path = os.path.join(base_log_path, EXP_NAME)

def to_numpy(x):
    return x.detach().cpu()


def mean_patk(output, multilabel_onehot_target, k=1):
    output = to_numpy(output)
    multilabel_onehot_target = to_numpy(multilabel_onehot_target)
    mean_patk_metric = np.mean([len(np.intersect1d(torch.topk(output[b, :], k=k, dim=0).indices.numpy(),
                                                   torch.where(multilabel_onehot_target[b, :] == 1)[0].numpy())) / k
                                for b in range(output.shape[0])])
    return mean_patk_metric


def mean_ratk(output, multilabel_onehot_target, k=1):
    output = to_numpy(output)
    multilabel_onehot_target = to_numpy(multilabel_onehot_target)
    mean_ratk_metric = np.mean([len(np.intersect1d(torch.topk(output[b, :], k=k, dim=0).indices.numpy(),
                                                   torch.where(multilabel_onehot_target[b, :] == 1)[0].numpy())) /
                                len(torch.where(multilabel_onehot_target[b, :] == 1)[0].numpy())
                                for b in range(output.shape[0])])
    return mean_ratk_metric


def mapk(output, multilabel_onehot_target, k=1):
    output = to_numpy(output)
    multilabel_onehot_target = to_numpy(multilabel_onehot_target)
    mapk_metric = np.mean([np.mean([len(np.intersect1d(torch.topk(output[b, :], k=i, dim=0).indices.numpy(),
                                                      torch.where(multilabel_onehot_target[b, :] == 1)[0].numpy())) / i
                                    for i in range(1, k+1)]) for b in range(output.shape[0])])
    return mapk_metric


data_shapes = {
    'numerical' : cat_vocab_size + 1,
    'categorical' : [dt_vocab_size, id_vocab_size]
}
model = Baseline(data_shapes)
optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=scheduler_factor, patience=scheduler_patience)
criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')


runner = dl.SupervisedRunner(
    device=device,
    input_key=['numerical', 'categorical'],
    output_key=None,
    
)

runner.train(
    model=model.to(device),  
    criterion=criterion,
    optimizer=optimizer,    
    loaders={
        'train':train_dataloader,
        'valid':valid_dataloader,
        'test':test_dataloader
    },    
    verbose=True,
    main_metric='mean_pat_1',
    scheduler=scheduler,
    minimize_metric=False,
    initial_seed=seed,
    callbacks=[ 
        dl.LoaderMetricCallback('mean_pat_1', lambda o,i: mean_patk(o,i,1)),
        dl.LoaderMetricCallback('mean_rat_1', lambda o,i: mean_ratk(o,i,1)),
        dl.LoaderMetricCallback('mapk_1', lambda o,i: mapk(o,i,1)),
        
        dl.LoaderMetricCallback('mean_pat_2', lambda o,i: mean_patk(o,i,2)),
        dl.LoaderMetricCallback('mean_rat_2', lambda o,i: mean_ratk(o,i,2)),
        dl.LoaderMetricCallback('mapk_2', lambda o,i: mapk(o,i, 2)),
        
        dl.LoaderMetricCallback('mean_pat_3', lambda o,i: mean_patk(o,i,3)),
        dl.LoaderMetricCallback('mean_rat_3', lambda o,i: mean_ratk(o,i,3)),
        dl.LoaderMetricCallback('mapk_3', lambda o,i: mapk(o,i, 3)),
        
        dl.LoaderMetricCallback('mean_pat_4', lambda o,i: mean_patk(o,i, 4)),
        dl.LoaderMetricCallback('mean_rat_4', lambda o,i: mean_ratk(o,i, 4)),
        dl.LoaderMetricCallback('mapk_4', lambda o,i: mapk(o,i, 4)),
    ],    
    logdir=exp_log_path,    
    num_epochs=500,
    load_best_on_end=True
)

