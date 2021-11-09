# Time pred
from data_preparation.window_split import window_combination
from data_preparation.dataset_preparation import OrderDataset
import numpy as np
from sklearn.metrics import r2_score
import json
from sacred import Experiment
import os
import pandas as pd
import statistics


def mode(array):
    most = max(list(map(array.count, array)))
    return list(set(filter(lambda x: array.count(x) == most, array)))

def baseline_model(data_folder, train_file, test_file, look_back, init_data=True):

    if init_data:
        model_name = 'Baseline_model_in_time'
        train_data, test_data = OrderDataset(data_folder, train_file, test_file, look_back).prepare_data()
    else:
        model_name = 'Baseline_model_in_time_with_difficult_ship'
        train_data = pd.read_csv(os.path.join(data_folder, train_file))
        test_data = pd.read_csv(os.path.join(data_folder, test_file))
        for df in [train_data, test_data]:
            df.drop(['Unnamed: 0'], axis=1, inplace=True)
    test_comb = window_combination(test_data, look_back)
    y_pred = []
    y_gt = []
    for (ref_points, pred_point) in test_comb:
        y_pred.append(np.mean(test_data.loc[ref_points, 'dt'].values))
        #y_pred.append(mode(test_data.loc[ref_points, 'dt'].values.tolist())[0])
        y_gt.append(test_data.loc[pred_point, 'dt'])
    r2_metric = r2_score(y_gt, y_pred)

    print(f'{model_name}, test_r2_score = {r2_metric}')
    os.makedirs('results/', exist_ok=True)
    with open(f'results/{model_name}_metrics.json', 'w', encoding='utf-8') as f:
        json.dump({'test_r2_score': r2_metric}, f)