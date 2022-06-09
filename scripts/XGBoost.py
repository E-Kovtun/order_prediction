import torch
import sys
sys.path.append("../",)

#from launch_scripts.train_lstm_material import mean_patk, mean_ratk, mapk
from data_preparation.data_reader import OrderReader
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
import os
import json
from tqdm import tqdm
import xgboost as xgb
from utils.roc_auc_metrics import calculate_roc_auc_metrics
from sklearn.multioutput import MultiOutputClassifier
import pickle as pkl

torch.manual_seed(42)
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

    look_back = 3

    prepared_folder = "datasets/data/prepared/order/"
    batch_size = 32
    dataloader_num_workers = 2
    rand_seed_list = [1, 2, 3, 4, 5]
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    all_roc_auc_micro = []
    all_roc_auc_macro = []
    for rand_seed in tqdm(rand_seed_list):
        torch.manual_seed(rand_seed)

        # data_folder = "/NOTEBOOK/UHH/Repository/Repository_LSTM/"
        # train_file = "df_beer_train_nn.csv"
        # train_test_file = "train_test.csv"
        # test_file = "df_beer_test.csv"
        # valid_file = "df_beer_valid_nn.csv"
        model_name = 'XGBoost_multilabel'
        dataset_name = os.path.basename(os.path.normpath(prepared_folder))
        os.makedirs(os.path.join('checkpoints/', dataset_name, model_name), exist_ok=True)
        checkpoint = os.path.join('checkpoints/', dataset_name, model_name,
                                  f'checkpoint_look_back_{look_back}_seed_{rand_seed}.pt')

        train_dataset = OrderReader(prepared_folder, look_back, 'train')
        test_dataset = OrderReader(prepared_folder, look_back, 'test')
        # valid_dataset = OrderReader(prepared_folder, 'valid')
        train_test_dataset = OrderReader(prepared_folder, look_back, 'test')

        # model_name = 'XGBoost_multilabel'
        # results_folder = f'../XGBoost_multilabel/{model_name}/'
        # checkpoint = results_folder + f'checkpoints/look_back_{look_back}_xgboost.pt'

        #os.makedirs(results_folder+'checkpoints/', exist_ok=True)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=dataloader_num_workers)
        entire_batch_cat_arr = []
        entire_batch_current_cat = []
        entire_batch_dt_arr = []
        entire_batch_amount_arr = []
        entire_batch_id_arr = []
        for batch_ind, batch_arrays in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            batch_arrays = [arr.to(device) for arr in batch_arrays]
            [batch_cat_arr, batch_current_cat, batch_dt_arr, batch_amount_arr, batch_id_arr] = batch_arrays
            entire_batch_cat_arr.append(batch_cat_arr)
            entire_batch_current_cat.append(batch_current_cat)
            entire_batch_dt_arr.append(batch_dt_arr)
            entire_batch_amount_arr.append(batch_amount_arr)
            entire_batch_id_arr.append(batch_id_arr)

        one_hot_id = F.one_hot(entire_batch_id_arr, train_dataset.id_vocab_size)
        print(one_hot_id)
        one_hot_label = F.one_hot(entire_batch_current_cat, train_dataset.cat_vocab_size)
        X_train = torch.cat((one_hot_id,
                             torch.cat((one_hot_label, entire_batch_amount_arr),
                                       axis=2).reshape(len(one_hot_id), -1)), axis=1)

        # create XGBoost instance with default hyper-parameters
        xgb_estimator = xgb.XGBClassifier(objective='binary:logistic')

        # create MultiOutputClassifier instance with XGBoost model inside
        multilabel_model = MultiOutputClassifier(xgb_estimator)

        # fit the model
        print('Training...')
        multilabel_model.fit(X_train, train_dataset.onehot_current_cat)
        pkl.dump(multilabel_model, open(results_folder+f"checkpoints/look_back_{look_back}_xgboost_multilabel.pkl", 'wb'))

    #------------------------------------------------------------
        # model_name = 'XGBoost_multilabel'
        # results_folder = f'../XGBoost_multilabel/{model_name}/'
        # checkpoint = results_folder + f'checkpoints/look_back_{look_back}_xgboost_multilabel.pkl'
        # multilabel_model = pkl.load(open(checkpoint, 'rb'))
        test_len = len(test_dataset.id_arr)
        X_test = torch.cat((train_test_dataset.id_arr[-test_len:],
                            torch.cat((train_test_dataset.mask_cat[-test_len:] * train_test_dataset.cat_arr[-test_len:],
                                       train_test_dataset.num_arr[-test_len:]),
                                      axis=2).reshape(len(train_test_dataset.id_arr[-test_len:]), -1)), axis=1)

        # all_output = torch.Tensor(multilabel_model.predict(X_test))
        # all_gt = train_test_dataset.onehot_current_cat[-test_len:]
        # test_mean_patk = {i: mean_patk(all_output, all_gt, k=i) for i in range(1, 5)}
        # test_mean_ratk = {i: mean_ratk(all_output, all_gt, k=i) for i in range(1, 5)}
        # test_mapk = {i: mapk(all_output, all_gt, k=i) for i in range(1, 5)}
        #
        # print(f'Test Precision@k {test_mean_patk} || Test Recall@k {test_mean_ratk}|| Test MAP@k {test_mapk})')
        #
        # with open(results_folder + f'{os.path.splitext(os.path.basename(checkpoint))[0]}.json', 'w', encoding='utf-8') as f:
        #     json.dump({'test_precision@k': test_mean_patk, 'test_recall@k': test_mean_ratk, 'test_map@k': test_mapk}, f)
        metrics_dict = calculate_roc_auc_metrics(np.array(all_scores), np.array(all_gt))
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


if __name__ == '__main__':
    train()