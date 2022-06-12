import torch
import sys
sys.path.append("../",)

#from launch_scripts.train_lstm_material import mean_patk, mean_ratk, mapk
from data_preparation.data_reader import OrderReader
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from utils.multilabel_metrics import calculate_all_metrics
import os
import json
from tqdm import tqdm
import xgboost as xgb
from utils.roc_auc_metrics import calculate_roc_auc_metrics
from sklearn.multioutput import MultiOutputClassifier
import pickle as pkl
import pandas as pd
import random


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

    prepared_folder = "datasets/data/prepared/demand/"
    dataset_name = 'demand'
    batch_size = 512
    dataloader_num_workers = 0
    rand_seed_list = [1]
    model_name = 'XGBoost_multilabel'
    all_roc_auc_micro = []
    all_roc_auc_macro = []
    for rand_seed in tqdm(rand_seed_list):
        # random.seed(rand_seed)
        # print(torch.manual_seed(rand_seed))
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        model_name = 'XGBoost_multilabel'
        dataset_name = os.path.basename(os.path.normpath(prepared_folder))
        os.makedirs(os.path.join('checkpoints/', dataset_name, model_name), exist_ok=True)
        checkpoint = os.path.join('checkpoints/', dataset_name, model_name,
                                  f'checkpoint_look_back_{look_back}_seed_{rand_seed}.pkl')

        train_dataset = OrderReader(prepared_folder, look_back, 'train')


        #os.makedirs(results_folder+'checkpoints/', exist_ok=True)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=dataloader_num_workers)
        # FOR HUGE DATASETS
        # for split in range(8):
        # break
        entire_batch_cat_arr = []
        entire_batch_current_cat = []
        entire_batch_dt_arr = []
        entire_batch_amount_arr = []
        entire_batch_id_arr = []
        for batch_ind, batch_arrays in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # if batch_ind < len(train_dataloader) * split // 8:
            #     continue
            [batch_cat_arr, batch_current_cat, batch_dt_arr, batch_amount_arr, batch_id_arr] = batch_arrays
            entire_batch_cat_arr.append(batch_cat_arr)
            entire_batch_current_cat.append(batch_current_cat)
            entire_batch_dt_arr.append(batch_dt_arr)
            entire_batch_amount_arr.append(batch_amount_arr)
            entire_batch_id_arr.append(batch_id_arr)
            # if batch_ind > len(train_dataloader) * (split + 1) // 8:
            #     break

        #ID ONE-HOT
        entire_batch_id_arr = torch.cat(entire_batch_id_arr, 0) # [sample_size]
        one_hot_id = F.one_hot(entire_batch_id_arr, train_dataset.id_vocab_size) # [sample_size, id_vocab_size]
        entire_batch_id_arr = []
        #LABEL ONE-HOT
        entire_batch_cat_arr = torch.cat(entire_batch_cat_arr, 0) # [sample_size, look_back, max_cat_len]
        mask_cat = torch.tensor(~(entire_batch_cat_arr == train_dataset.cat_vocab_size), dtype=torch.int64) # [sample_size, look_back, max_cat_len]
        one_hot_label = torch.sum(F.one_hot(entire_batch_cat_arr, train_dataset.cat_vocab_size+1) * mask_cat.unsqueeze(3), dim=2)[:, :, :-1]
        entire_batch_cat_arr = []
        # [sample_size, look_back, cat_vocab_size]
        # one_hot_label = torch.sum(one_hot_label, dim=1)
        # # [sample_size, cat_vocab_size]

        #AMOUNT ONE-HOT
        entire_batch_amount_arr = torch.cat(entire_batch_amount_arr, 0) # [sample_size, look_back, max_cat_len]
        one_hot_amount = torch.sum(F.one_hot(entire_batch_amount_arr, train_dataset.amount_vocab_size+1) * mask_cat.unsqueeze(3), dim=2)[:, :, :-1]
        entire_batch_amount_arr = []
        # [sample_size, look_back, amount_vocab_size]
        # one_hot_amount = torch.sum(one_hot_amount, dim=1)
        # # [sample_size, amount_vocab_size]

        entire_batch_dt_arr = torch.cat(entire_batch_dt_arr, 0) # [sample_size]

        #CONCATENATE ID, dt, LABEL, AMOUNT
        sample_size = one_hot_id.shape[0]
        print(one_hot_id.unsqueeze(1).shape)

        X_train = torch.cat((one_hot_id.squeeze(1),
                             one_hot_label.reshape(sample_size, look_back*train_dataset.cat_vocab_size),
                             one_hot_amount.reshape(sample_size, look_back*train_dataset.amount_vocab_size),
                             entire_batch_dt_arr.squeeze(1)), dim=1) # [sample_size, num_of_features]


        # fit the model
        print('Training...')
        # create XGBoost instance with default hyper-parameters
        print(rand_seed)
        xgb_estimator = xgb.XGBClassifier(objective='binary:logistic', seed=rand_seed)

        # create MultiOutputClassifier instance with XGBoost model inside
        multilabel_model = MultiOutputClassifier(xgb_estimator)

        #CREATE VECTOR OF LABELS FOR CURRENT DATE
        entire_batch_current_cat = torch.cat(entire_batch_current_cat, 0) # [sample_size, max_cat_len]
        mask_current_cat = torch.tensor(~(entire_batch_current_cat == train_dataset.cat_vocab_size), dtype=torch.int64).unsqueeze(2)
        # [sample_size, max_cat_len, 1]

        onehot_current_cat = torch.sum(F.one_hot(entire_batch_current_cat, num_classes=train_dataset.cat_vocab_size + 1) * mask_current_cat,
                                       dim=1)[:, :-1] # [sample_size, cat_vocab_size]

        multilabel_model.fit(X_train, onehot_current_cat)

        pkl.dump(multilabel_model, open(checkpoint, 'wb')) #results_folder+f"checkpoints/look_back_{look_back}_xgboost_multilabel.pkl", 'wb'))

    #------------------------------------------------------------
        # model_name = 'XGBoost_multilabel'
        # results_folder = f'../XGBoost_multilabel/{model_name}/'
        # checkpoint = os.path.join('checkpoints/', dataset_name, model_name,
        #                           f'checkpoint_look_back_{look_back}_seed_1.pkl')
        # multilabel_model = pkl.load(open(checkpoint, 'rb'))

        # For to provide some metrics with thresholds

        # print('Validation...')
        # valid_dataset = OrderReader(prepared_folder, look_back, 'test')
        # valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,
        #                              num_workers=dataloader_num_workers)
        # entire_batch_cat_arr_valid = []
        # entire_batch_current_cat_valid = []
        # entire_batch_dt_arr_valid = []
        # entire_batch_amount_arr_valid = []
        # entire_batch_id_arr_valid = []
        # for batch_ind, batch_arrays in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader)):
        #     [batch_cat_arr, batch_current_cat, batch_dt_arr, batch_amount_arr, batch_id_arr] = batch_arrays
        #     entire_batch_cat_arr_valid.append(batch_cat_arr)
        #     entire_batch_current_cat_valid.append(batch_current_cat)
        #     entire_batch_dt_arr_valid.append(batch_dt_arr)
        #     entire_batch_amount_arr_valid.append(batch_amount_arr)
        #     entire_batch_id_arr_valid.append(batch_id_arr)
        #
        # # ID ONE-HOT
        # entire_batch_id_arr_valid = torch.cat(entire_batch_id_arr_valid, 0)  # [sample_size]
        # one_hot_id_valid = F.one_hot(entire_batch_id_arr_valid,
        #                             valid_dataset.id_vocab_size)  # [sample_size, id_vocab_size]
        #
        # # LABEL ONE-HOT
        # entire_batch_cat_arr_valid = torch.cat(entire_batch_cat_arr_valid, 0)  # [sample_size, look_back, max_cat_len]
        # mask_cat_valid = torch.tensor(~(entire_batch_cat_arr_valid == valid_dataset.cat_vocab_size),
        #                              dtype=torch.int64)  # [sample_size, look_back, max_cat_len]
        # one_hot_label_valid = torch.sum(
        #     F.one_hot(entire_batch_cat_arr_valid, valid_dataset.cat_vocab_size + 1) * mask_cat_valid.unsqueeze(3), dim=2)[
        #                      :, :, :-1]
        # # [sample_size, look_back, cat_vocab_size]
        # # one_hot_label_valid = torch.sum(one_hot_label_valid, dim=1)
        # # # [sample_size, cat_vocab_size]
        #
        # # AMOUNT ONE-HOT
        # entire_batch_amount_arr_valid = torch.cat(entire_batch_amount_arr_valid,
        #                                          0)  # [sample_size, look_back, max_cat_len]
        # one_hot_amount_valid = torch.sum(
        #     F.one_hot(entire_batch_amount_arr_valid, valid_dataset.amount_vocab_size + 1) * mask_cat_valid.unsqueeze(3),
        #     dim=2)[:, :, :-1]
        # # [sample_size, look_back, amount_vocab_size]
        # # one_hot_amount_valid = torch.sum(one_hot_amount_valid, dim=1)
        # # # [sample_size, amount_vocab_size]
        #
        # entire_batch_dt_arr_valid = torch.cat(entire_batch_dt_arr_valid, 0)  # [sample_size]
        #
        # # CONCATENATE ID, dt, LABEL, AMOUNT
        # sample_size = one_hot_id_valid.shape[0]
        # X_valid = torch.cat((one_hot_id_valid.squeeze(1),
        #                     one_hot_label_valid.reshape(sample_size, look_back * train_dataset.cat_vocab_size),
        #                     one_hot_amount_valid.reshape(sample_size, look_back * train_dataset.amount_vocab_size),
        #                     entire_batch_dt_arr_valid.squeeze(1)), dim=1)  # [sample_size, num_of_features]
        #
        # conf_scores_valid = torch.Tensor(multilabel_model.predict(X_valid))
        # entire_batch_current_cat_valid = torch.cat(entire_batch_current_cat_valid, 0)
        # mask_cat_valid = torch.tensor(~(entire_batch_current_cat_valid == valid_dataset.cat_vocab_size),
        #                              dtype=torch.int64)
        # all_gt_valid = torch.sum(F.one_hot(entire_batch_current_cat_valid) * mask_cat_valid.unsqueeze(2), dim=1)[:,
        #          :-1]  # [sample_size, cat_vocab_size]

        print('Testing ...')
        test_dataset = OrderReader(prepared_folder, look_back, 'test')
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=dataloader_num_workers)
        entire_batch_cat_arr_test = []
        entire_batch_current_cat_test = []
        entire_batch_dt_arr_test = []
        entire_batch_amount_arr_test = []
        entire_batch_id_arr_test = []
        for batch_ind, batch_arrays in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            [batch_cat_arr, batch_current_cat, batch_dt_arr, batch_amount_arr, batch_id_arr] = batch_arrays
            entire_batch_cat_arr_test.append(batch_cat_arr)
            entire_batch_current_cat_test.append(batch_current_cat)
            entire_batch_dt_arr_test.append(batch_dt_arr)
            entire_batch_amount_arr_test.append(batch_amount_arr)
            entire_batch_id_arr_test.append(batch_id_arr)

        #ID ONE-HOT
        entire_batch_id_arr_test = torch.cat(entire_batch_id_arr_test, 0) # [sample_size]
        one_hot_id_test = F.one_hot(entire_batch_id_arr_test, test_dataset.id_vocab_size) # [sample_size, id_vocab_size]

        #LABEL ONE-HOT
        entire_batch_cat_arr_test = torch.cat(entire_batch_cat_arr_test, 0) # [sample_size, look_back, max_cat_len]
        mask_cat_test = torch.tensor(~(entire_batch_cat_arr_test == test_dataset.cat_vocab_size), dtype=torch.int64) # [sample_size, look_back, max_cat_len]
        one_hot_label_test = torch.sum(F.one_hot(entire_batch_cat_arr_test, test_dataset.cat_vocab_size+1) * mask_cat_test.unsqueeze(3), dim=2)[:, :, :-1]
        # [sample_size, look_back, cat_vocab_size]
        # one_hot_label_test = torch.sum(one_hot_label_test, dim=1)
        # # [sample_size, cat_vocab_size]

        #AMOUNT ONE-HOT
        entire_batch_amount_arr_test = torch.cat(entire_batch_amount_arr_test, 0) # [sample_size, look_back, max_cat_len]
        one_hot_amount_test = torch.sum(F.one_hot(entire_batch_amount_arr_test, test_dataset.amount_vocab_size+1) * mask_cat_test.unsqueeze(3), dim=2)[:, :, :-1]
        # [sample_size, look_back, amount_vocab_size]
        # one_hot_amount_test = torch.sum(one_hot_amount_test, dim=1)
        # # [sample_size, amount_vocab_size]

        entire_batch_dt_arr_test = torch.cat(entire_batch_dt_arr_test, 0) # [sample_size]

        #CONCATENATE ID, dt, LABEL, AMOUNT
        sample_size = one_hot_id_test.shape[0]
        X_test = torch.cat((one_hot_id_test.squeeze(1),
                             one_hot_label_test.reshape(sample_size, look_back*train_dataset.cat_vocab_size),
                             one_hot_amount_test.reshape(sample_size, look_back*train_dataset.amount_vocab_size),
                             entire_batch_dt_arr_test.squeeze(1)), dim=1) # [sample_size, num_of_features]

        conf_scores = torch.Tensor(multilabel_model.predict(X_test))
        entire_batch_current_cat_test = torch.cat(entire_batch_current_cat_test, 0)
        mask_cat_test = torch.tensor(~(entire_batch_current_cat_test == test_dataset.cat_vocab_size), dtype=torch.int64)
        all_gt = torch.sum(F.one_hot(entire_batch_current_cat_test) * mask_cat_test.unsqueeze(2), dim=1)[:, :-1] # [sample_size, cat_vocab_size]
        metrics_dict = calculate_roc_auc_metrics(np.array(conf_scores), np.array(all_gt))
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
        print(df)
        df.to_csv(os.path.join('results/', dataset_name, model_name, f'metrics_final.csv'))
        # metrics_dict = calculate_all_metrics(np.array(all_gt), np.array(conf_scores),
        #                                      np.array(all_gt_valid), np.array(conf_scores_valid), kind='thr')


        # test_mean_patk = {i: mean_patk(conf_scores, all_gt, k=i) for i in range(1, 5)}
        # test_mean_ratk = {i: mean_ratk(conf_scores, all_gt, k=i) for i in range(1, 5)}
        # test_mapk = {i: mapk(conf_scores, all_gt, k=i) for i in range(1, 5)}
        #
        # print(f'Test Precision@k {test_mean_patk} || Test Recall@k {test_mean_ratk}|| Test MAP@k {test_mapk})')



if __name__ == '__main__':
    train()

# import torch
# import sys
# sys.path.append("../",)
#
# #from launch_scripts.train_lstm_material import mean_patk, mean_ratk, mapk
# from data_preparation.data_reader import OrderReader
# from torch.utils.data import DataLoader
# import numpy as np
# import torch.nn.functional as F
# import os
# import json
# from tqdm import tqdm
# import xgboost as xgb
# from utils.roc_auc_metrics import calculate_roc_auc_metrics
# from sklearn.multioutput import MultiOutputClassifier
# import pickle as pkl
# import pandas as pd
# import random
#
#
# def mean_patk(output, multilabel_onehot_target, k):
#     mean_patk_metric = np.mean([len(np.intersect1d(torch.topk(output[b, :], k=k, dim=0).indices.numpy(),
#                                                    torch.where(multilabel_onehot_target[b, :] == 1)[0].numpy())) / k
#                                 for b in range(output.shape[0])])
#     return mean_patk_metric
#
#
# def mean_ratk(output, multilabel_onehot_target, k):
#     mean_ratk_metric = np.mean([len(np.intersect1d(torch.topk(output[b, :], k=k, dim=0).indices.numpy(),
#                                                    torch.where(multilabel_onehot_target[b, :] == 1)[0].numpy())) /
#                                 len(torch.where(multilabel_onehot_target[b, :] == 1)[0].numpy())
#                                 for b in range(output.shape[0])])
#     return mean_ratk_metric
#
#
# def mapk(output, multilabel_onehot_target, k):
#     mapk_metric = np.mean([np.mean([len(np.intersect1d(torch.topk(output[b, :], k=i, dim=0).indices.numpy(),
#                                                       torch.where(multilabel_onehot_target[b, :] == 1)[0].numpy())) / i
#                                     for i in range(1, k+1)]) for b in range(output.shape[0])])
#     return mapk_metric
#
# def train():
#
#     look_back = 3
#
#     prepared_folder = "datasets/data/prepared/gender/"
#     dataset_name = 'gender'
#     batch_size = 32
#     dataloader_num_workers = 0
#     rand_seed_list = [1, 2, 3, 4, 5]
#     model_name = 'XGBoost_multilabel'
#     all_roc_auc_micro = []
#     all_roc_auc_macro = []
#     for rand_seed in tqdm(rand_seed_list):
#         # random.seed(rand_seed)
#         # print(torch.manual_seed(rand_seed))
#         if torch.cuda.is_available():
#             device = torch.device('cuda:0')
#         else:
#             device = torch.device('cpu')
#         # data_folder = "/NOTEBOOK/UHH/Repository/Repository_LSTM/"
#         # train_file = "df_beer_train_nn.csv"
#         # train_test_file = "train_test.csv"
#         # test_file = "df_beer_test.csv"
#         # valid_file = "df_beer_valid_nn.csv"
#         model_name = 'XGBoost_multilabel'
#         dataset_name = os.path.basename(os.path.normpath(prepared_folder))
#         os.makedirs(os.path.join('checkpoints/', dataset_name, model_name), exist_ok=True)
#         checkpoint = os.path.join('checkpoints/', dataset_name, model_name,
#                                   f'checkpoint_look_back_{look_back}_seed_{rand_seed}.pkl')
#
#         train_dataset = OrderReader(prepared_folder, look_back, 'train')
#         test_dataset = OrderReader(prepared_folder, look_back, 'test')
#         # valid_dataset = OrderReader(prepared_folder, 'valid')
#         train_test_dataset = OrderReader(prepared_folder, look_back, 'test')
#
#         # model_name = 'XGBoost_multilabel'
#         results_folder = f'../XGBoost_multilabel/{model_name}/'
#         # checkpoint = results_folder + f'checkpoints/look_back_{look_back}_xgboost.pt'
#
#         #os.makedirs(results_folder+'checkpoints/', exist_ok=True)
#         train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
#                                       num_workers=dataloader_num_workers)
#         test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
#                                       num_workers=dataloader_num_workers)
#         entire_batch_cat_arr = []
#         entire_batch_current_cat = []
#         entire_batch_dt_arr = []
#         entire_batch_amount_arr = []
#         entire_batch_id_arr = []
#         for batch_ind, batch_arrays in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
#             [batch_cat_arr, batch_current_cat, batch_dt_arr, batch_amount_arr, batch_id_arr] = batch_arrays
#             entire_batch_cat_arr.append(batch_cat_arr)
#             entire_batch_current_cat.append(batch_current_cat)
#             entire_batch_dt_arr.append(batch_dt_arr)
#             entire_batch_amount_arr.append(batch_amount_arr)
#             entire_batch_id_arr.append(batch_id_arr)
#
#         #ID ONE-HOT
#         entire_batch_id_arr = torch.cat(entire_batch_id_arr, 0) # [sample_size]
#         one_hot_id = F.one_hot(entire_batch_id_arr, train_dataset.id_vocab_size) # [sample_size, id_vocab_size]
#
#         #LABEL ONE-HOT
#         entire_batch_cat_arr = torch.cat(entire_batch_cat_arr, 0) # [sample_size, look_back, max_cat_len]
#         mask_cat = torch.tensor(~(entire_batch_cat_arr == train_dataset.cat_vocab_size), dtype=torch.int64) # [sample_size, look_back, max_cat_len]
#         one_hot_label = torch.sum(F.one_hot(entire_batch_cat_arr, train_dataset.cat_vocab_size+1) * mask_cat.unsqueeze(3), dim=2)[:, :, :-1]
#         # [sample_size, look_back, cat_vocab_size]
#
#         #AMOUNT ONE-HOT
#         entire_batch_amount_arr = torch.cat(entire_batch_amount_arr, 0) # [sample_size, look_back, max_cat_len]
#         one_hot_amount = torch.sum(F.one_hot(entire_batch_amount_arr, train_dataset.amount_vocab_size+1) * mask_cat.unsqueeze(3), dim=2)[:, :, :-1]
#         # [sample_size, look_back, amount_vocab_size]
#
#         entire_batch_dt_arr = torch.cat(entire_batch_dt_arr, 0) # [sample_size]
#
#         #CONCATENATE ID, dt, LABEL, AMOUNT
#         sample_size = one_hot_id.shape[0]
#         print(one_hot_id.unsqueeze(1).shape)
#         print(one_hot_label.reshape(sample_size, look_back*train_dataset.cat_vocab_size).shape)
#         print(one_hot_amount.reshape(sample_size, look_back*train_dataset.amount_vocab_size).shape)
#         print(entire_batch_dt_arr.unsqueeze(1).shape)
#         X_train = torch.cat((one_hot_id.squeeze(1),
#                              one_hot_label.reshape(sample_size, look_back*train_dataset.cat_vocab_size),
#                              one_hot_amount.reshape(sample_size, look_back*train_dataset.amount_vocab_size),
#                              entire_batch_dt_arr.squeeze(1)), dim=1) # [sample_size, num_of_features]
#
#         # create XGBoost instance with default hyper-parameters
#         print(rand_seed)
#         xgb_estimator = xgb.XGBClassifier(objective='binary:logistic', seed=rand_seed)
#
#         # create MultiOutputClassifier instance with XGBoost model inside
#         multilabel_model = MultiOutputClassifier(xgb_estimator)
#
#         # fit the model
#         print('Training...')
#
#         #CREATE VECTOR OF LABELS FOR CURRENT DATE
#         entire_batch_current_cat = torch.cat(entire_batch_current_cat, 0) # [sample_size, max_cat_len]
#         mask_current_cat = torch.tensor(~(entire_batch_current_cat == train_dataset.cat_vocab_size), dtype=torch.int64).unsqueeze(2)
#         # [sample_size, max_cat_len, 1]
#
#         onehot_current_cat = torch.sum(F.one_hot(entire_batch_current_cat, num_classes=train_dataset.cat_vocab_size + 1) * mask_current_cat,
#                                        dim=1)[:, :-1] # [sample_size, cat_vocab_size]
#
#         multilabel_model.fit(X_train, onehot_current_cat)
#         pkl.dump(multilabel_model, open(checkpoint, 'wb')) #results_folder+f"checkpoints/look_back_{look_back}_xgboost_multilabel.pkl", 'wb'))
#
#     #------------------------------------------------------------
#         # model_name = 'XGBoost_multilabel'
#         # results_folder = f'../XGBoost_multilabel/{model_name}/'
#         # checkpoint = results_folder + f'checkpoints/look_back_{look_back}_xgboost_multilabel.pkl'
#         #multilabel_model = pkl.load(open(checkpoint, 'rb'))
#         # print('Testing ...')
#         entire_batch_cat_arr_test = []
#         entire_batch_current_cat_test = []
#         entire_batch_dt_arr_test = []
#         entire_batch_amount_arr_test = []
#         entire_batch_id_arr_test = []
#         for batch_ind, batch_arrays in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
#             [batch_cat_arr, batch_current_cat, batch_dt_arr, batch_amount_arr, batch_id_arr] = batch_arrays
#             entire_batch_cat_arr_test.append(batch_cat_arr)
#             entire_batch_current_cat_test.append(batch_current_cat)
#             entire_batch_dt_arr_test.append(batch_dt_arr)
#             entire_batch_amount_arr_test.append(batch_amount_arr)
#             entire_batch_id_arr_test.append(batch_id_arr)
#
#         #ID ONE-HOT
#         entire_batch_id_arr_test = torch.cat(entire_batch_id_arr_test, 0) # [sample_size]
#         one_hot_id_test = F.one_hot(entire_batch_id_arr_test, train_dataset.id_vocab_size) # [sample_size, id_vocab_size]
#
#         #LABEL ONE-HOT
#         entire_batch_cat_arr_test = torch.cat(entire_batch_cat_arr_test, 0) # [sample_size, look_back, max_cat_len]
#         mask_cat_test = torch.tensor(~(entire_batch_cat_arr_test == train_dataset.cat_vocab_size), dtype=torch.int64) # [sample_size, look_back, max_cat_len]
#         one_hot_label_test = torch.sum(F.one_hot(entire_batch_cat_arr_test, train_dataset.cat_vocab_size+1) * mask_cat_test.unsqueeze(3), dim=2)[:, :, :-1]
#         # [sample_size, look_back, cat_vocab_size]
#
#         #AMOUNT ONE-HOT
#         entire_batch_amount_arr_test = torch.cat(entire_batch_amount_arr_test, 0) # [sample_size, look_back, max_cat_len]
#         one_hot_amount_test = torch.sum(F.one_hot(entire_batch_amount_arr_test, train_dataset.amount_vocab_size+1) * mask_cat_test.unsqueeze(3), dim=2)[:, :, :-1]
#         # [sample_size, look_back, amount_vocab_size]
#
#         entire_batch_dt_arr_test = torch.cat(entire_batch_dt_arr_test, 0) # [sample_size]
#
#         #CONCATENATE ID, dt, LABEL, AMOUNT
#         sample_size = one_hot_id_test.shape[0]
#         X_test = torch.cat((one_hot_id_test.squeeze(1),
#                              one_hot_label_test.reshape(sample_size, look_back*train_dataset.cat_vocab_size),
#                              one_hot_amount_test.reshape(sample_size, look_back*train_dataset.amount_vocab_size),
#                              entire_batch_dt_arr_test.squeeze(1)), dim=1) # [sample_size, num_of_features]
#
#         conf_scores = torch.Tensor(multilabel_model.predict(X_test))
#         entire_batch_current_cat_test = torch.cat(entire_batch_current_cat_test, 0)
#         mask_cat_test = torch.tensor(~(entire_batch_current_cat_test == train_dataset.cat_vocab_size), dtype=torch.int64)
#         all_gt = torch.sum(F.one_hot(entire_batch_current_cat_test) * mask_cat_test.unsqueeze(2), dim=1)[:, :-1] # [sample_size, cat_vocab_size]
#
#         metrics_dict = calculate_roc_auc_metrics(np.array(conf_scores), np.array(all_gt))
#         print(calculate_roc_auc_metrics(np.array(conf_scores), np.array(all_gt)))
#         all_roc_auc_micro.append(metrics_dict['roc_auc_micro'])
#         all_roc_auc_macro.append(metrics_dict['roc_auc_macro'])
#         os.makedirs(os.path.join('results/', dataset_name, model_name), exist_ok=True)
#         with open(os.path.join('results/', dataset_name, model_name,
#                                f'metrics_look_back_{look_back}_seed_{rand_seed}.json'), 'w', encoding='utf-8') as f:
#             json.dump(metrics_dict, f)
#
#     df = pd.DataFrame(data=[[np.mean(all_roc_auc_micro), np.std(all_roc_auc_micro)],
#                                 [np.mean(all_roc_auc_macro), np.std(all_roc_auc_macro)]],
#                           index=['roc_auc_micro', 'roc_auc_macro'],
#                           columns=['mean', 'std'])
#     print(df)
#     df.to_csv(os.path.join('results/', dataset_name, model_name, f'metrics_final.csv'))
#         # one_hot_id_test = F.one_hot(entire_batch_id_arr_test, test_dataset.id_vocab_size)
#         # print(one_hot_id)
#         # one_hot_label_test = F.one_hot(entire_batch_cat_arr_test, test_dataset.cat_vocab_size + 1)
#         # X_test = torch.cat((one_hot_id_test,
#         #                      torch.cat((one_hot_label_test, entire_batch_amount_arr_test),
#         #                                axis=2).reshape(len(one_hot_id_test), -1)), axis=1)
#         # test_len = len(test_dataset.id_arr)
#         # X_test = torch.cat((train_test_dataset.id_arr[-test_len:],
#         #                     torch.cat((train_test_dataset.mask_cat[-test_len:] * train_test_dataset.cat_arr[-test_len:],
#         #                                train_test_dataset.num_arr[-test_len:]),
#         #                               axis=2).reshape(len(train_test_dataset.id_arr[-test_len:]), -1)), axis=1)
#
#         # all_gt.extend(batch_onehot_current_cat[:, :-1].detach().cpu().tolist())
#         # all_scores.extend(conf_scores.tolist())
#         # all_output = torch.Tensor(multilabel_model.predict(X_test))
#         # all_gt = train_test_dataset.onehot_current_cat[-test_len:]
#         # test_mean_patk = {i: mean_patk(all_output, all_gt, k=i) for i in range(1, 5)}
#         # test_mean_ratk = {i: mean_ratk(all_output, all_gt, k=i) for i in range(1, 5)}
#         # test_mapk = {i: mapk(all_output, all_gt, k=i) for i in range(1, 5)}
#         #
#         # print(f'Test Precision@k {test_mean_patk} || Test Recall@k {test_mean_ratk}|| Test MAP@k {test_mapk})')
#         #
#         # with open(results_folder + f'{os.path.splitext(os.path.basename(checkpoint))[0]}.json', 'w', encoding='utf-8') as f:
#         #     json.dump({'test_precision@k': test_mean_patk, 'test_recall@k': test_mean_ratk, 'test_map@k': test_mapk}, f)
#
#
#
# if __name__ == '__main__':
#     train()