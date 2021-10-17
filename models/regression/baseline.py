from data_preparation.window_split import window_combination
from data_preparation.dataset_preparation import OrderDataset
import numpy as np
from sklearn.metrics import r2_score
import json
from sacred import Experiment
import os


ex = Experiment('BaselineRegressor')
ex.add_config('configs/basic.json')
@ex.automain
def baseline_model(data_folder, train_file, test_file, look_back):
    model_name = 'BaselineRegressor'
    train_data, test_data = OrderDataset(data_folder, train_file, test_file, look_back).prepare_data()
    test_comb = window_combination(test_data, look_back)
    y_pred = []
    y_gt = []
    for (ref_points, pred_point) in test_comb:
        y_pred.append(np.mean(test_data.loc[ref_points, 'Amount_HL'].values))
        y_gt.append(test_data.loc[pred_point, 'Amount_HL'])

    r2_metric = r2_score(y_gt, y_pred)

    print(f'{model_name}, test_r2_score = {r2_metric}')
    os.makedirs('results/', exist_ok=True)
    with open(f'results/{model_name}_metrics.json', 'w', encoding='utf-8') as f:
        json.dump({'test_r2_score': r2_metric}, f)