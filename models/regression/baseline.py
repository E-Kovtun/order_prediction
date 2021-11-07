from data_preparation.dataset_preparation import OrderDataset
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
import json
from sacred import Experiment
import os


ex = Experiment('BaselineRegressor')
ex.add_config('configs/basic.json')
@ex.automain
def baseline_model(data_folder, train_file, test_file, look_back, fix_material, current_info, predicted_value):
    model_name = 'BaselineRegressor'
    order_dataset = OrderDataset(data_folder, train_file, test_file, look_back, fix_material, current_info, predicted_value)
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
