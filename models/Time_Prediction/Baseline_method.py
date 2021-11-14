from data_preparation.dataset_preparation import OrderDataset
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
import json
from sacred import Experiment
import os

'''
ex = Experiment('BaselineRegressor')
ex.add_config('configs/basic.json')
@ex.automain
'''
def baseline_model(data_folder, train_file, test_file, look_back, fix_material, current_info, predicted_value,
                   file_name, load_data=False):
    if load_data:
        order_dataset = OrderDataset(data_folder, train_file, test_file, look_back, fix_material, current_info,
                                     predicted_value, load_data)
        model_name = file_name
        train_data = pd.read_csv(os.path.join(data_folder, train_file))
        test_data = pd.read_csv(os.path.join(data_folder, test_file))
        for df in [train_data, test_data]:
            df.drop(['Unnamed: 0'], axis=1, inplace=True)
    else:
        model_name = file_name
        order_dataset = OrderDataset(data_folder, train_file, test_file, look_back, fix_material,
                                     current_info, predicted_value, load_data)
        train_data, test_data = order_dataset.preprocess_dataframe()

    test_comb = order_dataset.window_combinations(test_data)
    y_pred = []
    y_gt = []
    for (ref_points, pred_point) in test_comb:
        y_pred.append(np.mean(test_data.loc[ref_points, order_dataset.target_feature].values))
        y_gt.append(test_data.loc[pred_point, order_dataset.target_feature])

    r2_metric = r2_score(y_gt, y_pred)

    print(f'{model_name}, test_r2_score = {r2_metric}')
    os.makedirs('results/', exist_ok=True)
    with open(f'results/{model_name}_metrics.json', 'w', encoding='utf-8') as f:
        json.dump({'test_r2_score': r2_metric}, f)